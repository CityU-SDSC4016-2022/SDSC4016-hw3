from torch.utils.data import Dataset


class TwitterDataset(Dataset):
    def __init__(self, x_var: list, y_var: list):
        self.data = x_var
        self.label = y_var

    def __getitem__(self, idx: int):
        if self.label is None:
            return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
