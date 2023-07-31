from typing import Any
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data, window) -> None:
        super(Dataset,self).__init__()
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, index) -> Any:
        x = self.data[index : index + self.window]
        return x
