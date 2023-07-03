from torch.utils.data import Dataset, DataLoader


class Buffer(Dataset):
    def __init__(self, buffer):
        super(Buffer, self).__init__()
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


def get_loader(buffer: list, batch_size, shuffle=False):
    return DataLoader(
        Buffer(buffer), batch_size=batch_size, shuffle=shuffle
    )