import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from utility import fetch_sequences_from_fasta, construct_non_promoter_sequences, onehot_dna_sequences

class DataModuleClass(pl.LightningDataModule):
    
    def __init__(self, directory, batch_size, kfold):
        super().__init__()
        self.batch_size = batch_size
        self.dr = directory
        self.kf = kfold
        self.id = 0

    def set_current_id(self, id):
        self.id = id % self.kf 
    
    def prepare_data(self):    
        promoter_seg = fetch_sequences_from_fasta(path=self.dr, length=300)
        non_promoter_seg = construct_non_promoter_sequences(promoter_seg)
        promoter_seg_onhot = onehot_dna_sequences(promoter_seg)
        non_promoter_seg_onehot = onehot_dna_sequences(non_promoter_seg)
        ny = promoter_seg_onhot.shape[0]
        py = non_promoter_seg_onehot.shape[0]
        y_neg, y_pos = np.zeros(ny), np.ones(py)
        data = [np.vstack((promoter_seg_onhot, non_promoter_seg_onehot)), np.hstack((y_pos, y_neg))]
        self.dataset = shuffle(data[0], data[1], n_samples=ny+py, random_state=42)       
        
    def setup(self, stage=None): 
        skf = KFold(n_splits=self.kf, shuffle=True, random_state=42)
        skf.get_n_splits(self.dataset[0], self.dataset[1])
        self.folds = list(skf.split(self.dataset[0], self.dataset[1])) 

    def _shared_data_loder(self, val=False):
        X, y = self.dataset[0], self.dataset[1]
        if val: idx = self.folds[self.id][1]
        else: idx = self.folds[self.id][0]
        X, y = X[idx], y[idx]
        input_features = torch.as_tensor(data=X, dtype=torch.float)
        input_features = torch.transpose(input_features, 1, 2)
        y = torch.as_tensor(y, dtype=torch.float)
        return TensorDataset(input_features, y)
    
    def train_dataloader(self):
        self.train_data = self._shared_data_loder()
        return DataLoader(self.train_data,  batch_size = self.batch_size, shuffle=True) 
    
    def val_dataloader(self):
        self.valid_data = self._shared_data_loder(val=True)
        return DataLoader(self.valid_data, batch_size = self.batch_size, shuffle=False)