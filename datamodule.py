from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader

from dataclass import MorphologyDataSet


class MorphologyDataModule(LightningDataModule):
    def __init__(self, train, test, tokenizer, batch_size=8, max_x=50, max_y=50):

        super().__init__()
        self.train = train
        self.test = test
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_x = max_x
        self.max_y = max_y

    def setup(self, stage=None):
        self.train_dataset = MorphologyDataSet(self.train, self.tokenizer, self.max_x, self.max_y)
        self.test_dataset = MorphologyDataSet(self.test, self.tokenizer, self.max_x, self.max_y)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)
