from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
