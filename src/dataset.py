import torch
from torch.utils.data import Dataset

class SASRecDatasetOptimized(Dataset):
    def __init__(self, user_timelines_int, genre_tensor, year_tensor, max_len=200):
        self.user_ids = list(user_timelines_int.keys())
        self.timelines = user_timelines_int 
        self.genre_tensor = genre_tensor
        self.year_tensor = year_tensor
        self.max_len = max_len

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user = self.user_ids[idx]
        seq = self.timelines[user] 
        
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
            
        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            seq = [0] * pad_len + seq
            
        if len(seq) < 2:
             return (torch.zeros(self.max_len, dtype=torch.long),
                     torch.zeros(self.max_len, dtype=torch.long),
                     torch.zeros(self.max_len, dtype=torch.long),
                     torch.tensor(0, dtype=torch.long))
        
        target = seq[-1]
        input_list = seq[:-1] 
        input_list = [0] + input_list 
        
        input_tensor = torch.tensor(input_list, dtype=torch.long)
        genres = self.genre_tensor[input_tensor]
        years = self.year_tensor[input_tensor]
        
        return input_tensor, genres, years, torch.tensor(target, dtype=torch.long)