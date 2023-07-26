from torch.utils.data import Dataset
import torch
from torch import Tensor


class SMDDataLoader(Dataset):
    def __init__(self, filename, root_dir, transform):
        """
        Arguments:
            filename (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filename = filename
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, index: Tensor):
        if torch.is_tensor(index):
            index = index.tolist()


SMDDataLoader("test", "test", "test")
