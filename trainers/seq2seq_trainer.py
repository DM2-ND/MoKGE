import json
import os
import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm.auto import tqdm, trange

import numpy as np
import torch
from packaging import version
from torch import nn
from transformers import Trainer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from transformers.configuration_fsmt import FSMTConfig
from transformers.file_utils import WEIGHTS_NAME
from transformers.modeling_utils import PreTrainedModel

from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    TrainerState,
    TrainOutput,
    distributed_broadcast_scalars,
    distributed_concat,
    nested_concat,
    nested_numpify,
    set_seed,
)

from transformers.integrations import (
    is_optuna_available,
    is_tensorboard_available,
)

try:
    from transformers.utils import label_smoothed_nll_loss
except ImportError:
    from trainers.kgtrainer_utils import label_smoothed_nll_loss

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if is_tensorboard_available():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter

if is_optuna_available():
    import optuna

from evals.eval_acc_div import eval_accuracy_diversity
logger = logging.getLogger(__name__)


class Seq2SeqTrainer(Trainer):
    def __init__(self, config, data_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.data_args = data_args
        self.max_gen_length = data_args.val_max_target_length
        self.vocab_size = self.config.tgt_vocab_size if isinstance(self.config, FSMTConfig) else self.config.vocab_size

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            if self.args.sortish_sampler:
                self.train_dataset.make_sortish_sampler(
                    self.args.per_device_train_batch_size, distributed=self.args.n_gpu > 1
                )

            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    def _compute_loss(self, logits, labels):
        if self.args.label_smoothing == 0:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            assert logits.shape[-1] == self.vocab_size
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=self.config.pad_token_id
            )
        return loss

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():

            if self.args.predict_with_generate and not self.args.prediction_loss_only:
                num_return_sequences = self.data_args.eval_beams if self.data_args.do_sample else None
                expert_prompt = self.data_args.expert_prompt if hasattr(self.data_args, 'expert_prompt') else None

                generated_tokens = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                    num_beams=self.data_args.eval_beams,
                    num_return_sequences=num_return_sequences,
                    max_length=self.max_gen_length,
                    do_sample=self.data_args.do_sample,
                    top_k=self.data_args.top_k,
                    top_p=self.data_args.top_p,
                    expert_prompt=expert_prompt,
                )

                # in case the batch is shorter than max length, the output should be padded
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.max_gen_length)

            labels_out = inputs.get("labels")
            # Call forward again to get loss # TODO: avoidable?
            outputs = model(**inputs, use_cache=False)
            loss = self._compute_loss(outputs[1], labels_out)
            loss = loss.mean().item()
            if self.args.prediction_loss_only:
                return (loss, None, None)

            logits = generated_tokens if self.args.predict_with_generate else outputs[1]
        
        labels_out = self.repeat(labels_out, self.data_args.eval_beams)
        labels_out = labels_out.detach()
        labels = self._pad_tensors_to_max_len(labels_out, self.max_gen_length)
        return (loss, logits.detach(), labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        padded_tensor = self.config.pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):

        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        self.state = TrainerState()

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        # Check if a saved Trainer state exist
        if model_path is not None and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))

        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=not getattr(model.config, "gradient_checkpointing", False),
            )

        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        self.best_metric = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                self.total_flos = getattr(self._actual_model(model).config, "total_flos", 0)

                epochs_trained = self.global_step // num_update_steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % (num_update_steps_per_epoch)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Continuing training from %d non-embedding floating-point operations", self.total_flos)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.total_flos = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        model.zero_grad()
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    epoch_pbar.update(1)
                    continue

                tr_loss += self.training_step(model, inputs)
                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)
            
            if epoch < int(np.ceil(num_train_epochs)) and self.args.evaluate_during_training:

                output = self.evaluate()
                predictions = output.predictions.tolist()

                out_pred_path = os.path.join(self.args.output_dir, f'devout_epoch_{epoch+1}.txt')
                out_pred_metric = os.path.join(self.args.output_dir, f'devout_metric_{epoch+1}.json')

                with open(out_pred_path, 'w') as epoch_out:
                    for pred in predictions:
                        output_line = self.tokenizer.decode(pred, 
                            skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        epoch_out.write(output_line + '\n')
                
                metrics = {'epoch': epoch + 1}
                val_ref_path = os.path.join(self.args.data_dir, 'val.target')
                if not os.path.exists(val_ref_path):
                    val_ref_path += '.txt'
                metrics.update(eval_accuracy_diversity(out_pred_path, val_ref_path, self.args.eval_beams))

                with open(out_pred_metric, 'w') as metric_out:
                    json.dump(metrics, metric_out, indent=1)

                # ''' save the model '''
                if metrics[self.args.metric_for_best_model] > self.best_metric:
                    self.best_metric = metrics[self.args.metric_for_best_model]
                    self._save_training(model, trial, metrics=metrics)

            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(model, PreTrainedModel):
                self.model = model.from_pretrained(self.state.best_model_checkpoint)
                self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)

    def _save_training(self, model, trial, metrics=None):
        # In all cases (even distributed/parallel), self.model is always a reference
        # to the model we want to save.
        if hasattr(model, "module"):
            assert model.module is self.model, f"Module {model.module} should be a reference to self.model"
        else:
            assert model is self.model, f"Model {model} should be a reference to self.model"
        # Save model checkpoint
        checkpoint_folder = f"checkpoint_best_dev"
        if self.hp_search_backend is not None and trial is not None:
            run_id = trial.number if self.hp_search_backend == HPSearchBackend.OPTUNA else tune.get_trial_id()
            checkpoint_folder += f"-run-{run_id}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

        self.store_flos()
        self.save_model(output_dir)

        if self.is_world_process_zero():
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        # Determine the new best metric / best model checkpoint
        if metrics is not None:
            metric_to_check = self.args.metric_for_best_model
            metric_value = metrics[metric_to_check]
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.is_world_process_zero():
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Maybe delete some older checkpoints.
        if self.is_world_process_zero():
            self._rotate_checkpoints(use_mtime=True)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(eval_dataloader, description="Evaluation")
        return output

    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            if loss is not None:
                eval_losses.extend([loss] * batch_size)
            if logits is not None:
                preds = logits if preds is None else nested_concat(preds, logits, dim=0)
            if labels is not None:
                label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        
        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = nested_numpify(preds)
        if label_ids is not None:
            label_ids = nested_numpify(label_ids)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            if self.args.local_rank != -1:
                metrics["eval_loss"] = (
                    distributed_broadcast_scalars(eval_losses, num_total_examples=self.num_examples(dataloader)).mean().item())
            else:
                metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    @staticmethod
    def repeat(tensor, K):
        # [B, ...] => [B*K, ...] Used unsqueeze and transpose to avoid [K*B] when using torch.Tensor.repeat
        if isinstance(tensor, torch.Tensor):
            B, *size = tensor.size()
            expand_size = B, K, *size
            tensor = tensor.unsqueeze(1).expand(*expand_size).contiguous().view(B * K, *size)
            return tensor
        elif isinstance(tensor, list):
            out = []
            for x in tensor:
                for _ in range(K):
                    out.append(x.copy())
            return out