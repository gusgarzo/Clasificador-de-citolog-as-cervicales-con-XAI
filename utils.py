import torch
from torch.utils import data

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)

#BINARY_MAP = {0: 1, 1: 1, 2: 1, 3: 0}  # benigna=0, resto=1 ESTO SI QUEREMOS CLASIFICACION BINARIA
#Clasificacion cuaternaria
BINARY_MAP = {0: 0, 1: 1, 2: 2, 3: 3}  # benigna, ascus, lsil, hsil

TARGET_NAMES = ['benign', 'ascus', 'lsil', 'hsil']

class CitologiaDataset(data.Dataset):
    def __init__(self, original_dataset, binary_map):
        self.original_dataset = original_dataset
        self.binary_map = binary_map

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        return img, self.binary_map[label]