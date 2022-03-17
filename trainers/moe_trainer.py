from operator import index
import torch
from torch import nn
from typing import Any, Dict, Union
from trainers.seq2seq_trainer import Seq2SeqTrainer
from packaging import version

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


class MoESeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixtures = self.data_args.mixtures
        self.expert_prompt = self.data_args.expert_prompt
        self.mixture_embedding = self.data_args.mixture_embedding

    def _training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer) -> torch.Tensor:

        self.B, self.L = inputs['labels'].shape
        self.pad_mask = (inputs['labels'] == self.config.pad_token_id).view(self.B, 1, self.L).to(self.args.device)

        inputs = self._prepare_inputs(inputs)

        if self.mixture_embedding:
            mixture_ids = torch.arange(self.mixtures, dtype=torch.long, device=inputs['input_ids'].device).view(
                        self.mixtures, 1).repeat(inputs['input_ids'].shape)
            mixture_inputs = {k: self.repeat(v, self.mixtures) for k, v in inputs.items()}
            mixture_inputs['mixture_ids'] = mixture_ids
            model.eval()
            mixture_ids = self.compute_mixture_ids(model, mixture_inputs)
            inputs['mixture_ids'] = mixture_ids.expand(inputs['input_ids'].shape)

        else: # using prompt as different expert
            mixture_ids_prompt = self.expert_prompt.repeat(inputs['input_ids'].shape[0], 1).to(self.args.device)
            mixture_att_prompt = torch.full(mixture_ids_prompt.shape, 1).to(self.args.device)

            mixture_inputs = {k: self.repeat(v, self.mixtures) for k, v in inputs.items()}
            mixture_inputs['input_ids'] = torch.cat([mixture_ids_prompt, mixture_inputs['input_ids']], dim=1)
            mixture_inputs['attention_mask'] = torch.cat([mixture_att_prompt, mixture_inputs['attention_mask']], dim=1)

            model.eval()
            mixture_ids = self.compute_mixture_ids(model, mixture_inputs)
            expanded_mixture_ids = mixture_ids.expand(self.B, self.data_args.prompt_nums).unsqueeze(dim=1)
            input_ids_prompt = torch.gather(mixture_ids_prompt.view(
                self.B, self.mixtures, -1), dim=1, index=expanded_mixture_ids).squeeze()
            attention_prompt = torch.full(input_ids_prompt.shape, 1).to(self.args.device)
            inputs['input_ids'] = torch.cat([input_ids_prompt, inputs['input_ids']], dim=1)
            inputs['attention_mask'] = torch.cat([attention_prompt, inputs['attention_mask']], dim=1)

        # do the expert training!
        model.train()
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs):

        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        logits = outputs[0]
        return self._compute_loss(logits, labels)

    def compute_mixture_ids(self, model, inputs):
        
        _inputs = inputs.copy()
        _labels = _inputs.pop("labels")
        outputs = model(**_inputs, use_cache=False)
        logits = outputs[0]

        mixture_ids = self._compute_mixture_loss(logits, _labels)
        return mixture_ids

    def _compute_mixture_loss(self, logits, labels):

        assert logits.shape[:2] == labels.shape
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='none')
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1)).reshape(self.B, self.mixtures, self.L)
        mixture_ids = loss.masked_fill(self.pad_mask, 0).sum(dim=2).argmin(dim=1).unsqueeze(dim=1).type(torch.int64)

        return mixture_ids