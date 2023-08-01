from typing import Any
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data) -> None:
        super(Dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        return self.data[index]
