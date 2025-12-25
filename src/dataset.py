import torch
from torch.utils.data import Dataset

class SASRecDatasetSeq(Dataset):
    def __init__(self, inputs, genres, years, targets):
        self.inputs = inputs
        self.genres = genres
        self.years = years
        self.targets = targets
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.genres[idx], self.years[idx], self.targets[idx]