import torch
import numpy as np
from torch.utils.data import Dataset

class DiagnosisDataset(Dataset):  
  def __init__(self, input_ids, att_mask):
        self.input_ids = input_ids
        self.att_mask = att_mask
        
  def __len__(self):    
        return len(self.input_ids)

  def __getitem__(self, index):  
        input_id = torch.from_numpy(self.input_ids[index,:]).int()
        mask = torch.from_numpy(self.att_mask[index,:]).bool()  
        return input_id, mask

