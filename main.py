import pytorch_lightning as pl
import torch
from absl import app, flags
from numpy import save
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartModel,
    BartTokenizerFast,
)

import utils.myutil as myutil
from dataclass import MorphologyDataSet
from datamodule import MorphologyDataModule
from model_class import InflectionModel

FLAGS = flags.FLAGS
flags.DEFINE_string("language_task", "morphology", "folder to find data")
flags.DEFINE_string("language", "tur", "low resource language")
flags.DEFINE_integer("epochs", 6, "n_epochs")


def main(argv):
    train_path = f"data_2021/{FLAGS.language}/{FLAGS.language}.train"
    dev_path = f"data_2021/{FLAGS.language}/{FLAGS.language}.dev"
    file_path, special_tokens = myutil.make_data_for_tokenizer(
        train_path, dev_path, FLAGS.language
    )
    tokenizer = ByteLevelBPETokenizer()
    # Train the tokenizer
    file_path = "generated_data/data.txt"
    tokenizer.train(
        files=file_path, vocab_size=50_000, min_frequency=1, special_tokens=special_tokens
    )
    tokenizer.save_model(f"bart_local_{FLAGS.language}")
    tokenizer = BartTokenizerFast.from_pretrained("bart_local_tur")

    train_input, train_output, train_tags = myutil.read_data(train_path)
    validate_input, validate_output, validate_tags = myutil.read_data(dev_path)
    train, test, max_x, max_y = myutil.generate_data(
        train_input,
        train_tags,
        validate_input,
        validate_tags,
        train_output,
        validate_output,
    )
    print(train[:3])

    config = BartConfig(
        max_position_embedding=128,
        encoder_layers=3,
        encoder_attention_heads=4,
        decoder_layers=3,
        decoder_attention_heads=4,
        vocab_size=tokenizer.vocab_size,
    )

    model_to_pass = BartForConditionalGeneration(config)

    data_module = MorphologyDataModule(train, test, tokenizer, 2, max_x, max_y)

    model = InflectionModel(model_to_pass)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checks/checkpoints",
        filename="best",
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=6,
        gpus=1,
        progress_bar_refresh_rate=10,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    app.run(main)
