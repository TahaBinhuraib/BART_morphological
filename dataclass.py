import torch
from torch.utils.data import Dataset


class MorphologyDataSet(Dataset):
    def __init__(self, data, tokenizer, max_x, max_y):

        self.tokenizer = tokenizer
        self.data = data
        self.max_x = max_x
        self.max_y = max_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        input = data_point[0]
        output = data_point[1]
        text_encoding = self.tokenizer(
            input,
            max_length=self.max_x,
            padding="max_length",
            add_special_tokens=True,
        )

        output_encoding = self.tokenizer(
            output,
            max_length=self.max_y,
            padding="max_length",
            add_special_tokens=True,
        )
        labels_ = output_encoding["input_ids"]
        return dict(
            text=input,
            output_text=output,
            text_input_ids=torch.tensor([text_encoding["input_ids"]]).flatten(),
            text_attention_mask=torch.tensor([text_encoding["attention_mask"]]).flatten(),
            labels=torch.tensor([labels_]).flatten(),
            labels_attention_mask=torch.tensor([output_encoding["attention_mask"]]).flatten(),
        )
