import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from pytorch_lightning.metrics.classification import StatScores

n_lstm_layer = 1

class model(pl.LightningModule):
    
    def __init__(self, hidden_size, learning_rate, gru):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gru= gru
        self.criterion = nn.BCELoss()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=27) 
        self.pool1 = nn.MaxPool1d(kernel_size=6, stride=1)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=14, padding=9)
        self.pool2 = nn.MaxPool1d(kernel_size=6, stride=1)
        self.drop2 = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=7)
        self.pool3 = nn.MaxPool1d(kernel_size=6, stride=1)
        self.drop3 = nn.Dropout(p=0.5)
        if self.gru:
            self.bgru = nn.GRU(input_size=104, hidden_size=self.hidden_size, bidirectional=True, batch_first=True, num_layers=n_lstm_layer)  
        else:
            self.blstm = nn.LSTM(input_size=104, hidden_size=self.hidden_size, bidirectional=True, batch_first=True, num_layers=n_lstm_layer) 
        self.drop4 = nn.Dropout(p=0.5)        
        self.fc1 = nn.Linear(in_features=self.hidden_size*2, out_features=128)
        self.drop5 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x1 = F.pad(x, pad=(16, 15)) 
        cnn1 = F.relu(self.conv1(x1)) 
        cnn1 = self.pool1(cnn1)
        cnn1 = self.drop1(cnn1)
        cnn2 = F.relu(self.conv2(x))
        cnn2 = self.pool2(cnn2)
        cnn2 = self.drop2(cnn2)
        x3 = F.pad(x, pad=(6, 5))
        cnn3 = F.relu(self.conv3(x3))
        cnn3 = self.pool3(cnn3)
        cnn3 = self.drop3(cnn3)
        cnns = torch.cat((cnn1, cnn2, cnn3), dim=1) 
        cnns = torch.transpose(cnns, 1, 2) 
        h0 = torch.zeros(n_lstm_layer*2 , cnns.size(0) , self.hidden_size, device=self.device) 
        if not self.gru:
            c0 = torch.zeros(n_lstm_layer*2 , cnns.size(0) , self.hidden_size, device=self.device) 
            outputs, (h_n, c_n) = self.blstm(cnns, (h0,c0)) 
        else:
            outputs, h_n = self.bgru(cnns, h0) 
        h_n = torch.cat((h_n[0,:,:], h_n[1,:,:]), dim = 1) 
        d = self.drop4(h_n)
        c = F.relu(self.fc1(d))
        c = self.drop5(c)
        y = self.fc2(c)
        logits = torch.flatten(y)
        return torch.sigmoid(logits)
        

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss, y, y_hat

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalars("Loss", {"Train": avg_loss}, self.current_epoch)
                
    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step(batch)
        y = y.type(torch.int64)
        y_hat = y_hat.round().type(torch.int64)
        yc, y_hatc = y.cpu(), y_hat.cpu()
        tn, fp, fn, tp = confusion_matrix(yc, y_hatc, labels=[0,1]).ravel()
        precision = tp / (tp + fp)  
        if np.isnan(precision): precision = 0   
        recall = tp / (tp + fn)                                                                          
        if np.isnan(recall): recall = 0
        Mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))          
        if np.isnan(Mcc): Mcc = 0
        metrics = {'Precision': precision, 'Recall': recall}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('MCC', Mcc, on_step=False, on_epoch=True, prog_bar=True, logger=False) 
        return {'cost': loss, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['cost'] for x in outputs]).mean()
        sum_tn = np.stack([x['TN'] for x in outputs]).sum()
        sum_fp = np.stack([x['FP'] for x in outputs]).sum()
        sum_fn = np.stack([x['FN'] for x in outputs]).sum()
        sum_tp = np.stack([x['TP'] for x in outputs]).sum()
        self.logger.experiment.add_scalars("Confusion_matrix", {"TN": sum_tn, 'FP': sum_fp, 'FN': sum_fn, 'TP': sum_tp}, self.current_epoch)
        self.logger.experiment.add_scalars("Loss", {"Validation": avg_loss}, self.current_epoch)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)