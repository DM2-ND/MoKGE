from trainers.seq2seq_trainer import Seq2SeqTrainer

class VAESeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        lm_loss = self._compute_loss(outputs[0], labels)
        kl_loss = outputs[1]
        return lm_loss - kl_loss / 100 # alpha = 0.01
