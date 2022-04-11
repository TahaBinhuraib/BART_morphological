import torch
from pytorch_lightning import LightningModule
from transformers import AdamW, InputExample


class InflectionModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]

        loss, outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logged_loss = torch.tensor([loss])
        self.log("train_loss", logged_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]

        loss, outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logged_loss = torch.tensor([loss])
        self.log("val_loss", logged_loss, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001)
