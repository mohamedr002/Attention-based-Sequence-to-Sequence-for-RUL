import numpy as np
import torch
from data_processing import process_data
from torch.utils.data import TensorDataset, DataLoader
import math
from torch import nn

def load_data(data_identifier,window_size,batch_size,shuffle):

    data_dir= "/home/mohamed/raw_cmapps/"
    data_identifier=data_identifier;batch_size=batch_size ;shuffle_stats=shuffle;drop_last=True;window_size=window_size
    train_dl, test_dl=get_data(data_dir,data_identifier, batch_size,shuffle_stats,drop_last,window_size)
    dataloaders = { "train": train_dl, "validation": test_dl}
    return train_dl,test_dl,dataloaders
    
def get_data(data_dir,data_identifier, batch_size,shuffle_stats=True,drop_last=True,win_size=15):
    x_train, x_test,y_train,y_test = process_data(data_dir,data_identifier,win_size)
    x_train=(torch.from_numpy(x_train)).permute(0,2,1)
    y_train=torch.from_numpy(y_train)
    x_test=(torch.from_numpy(x_test)).permute(0,2,1)
    y_test= torch.DoubleTensor(y_test)
    
    if data_identifier=='FD002':
        x_test_bs= torch.cat((x_test,x_test[258,::].unsqueeze(0)))
        y_test_bs= torch.cat((y_test.view(259),y_test[-1].view(1)))    
    elif data_identifier=='FD004':
        x_test_bs= torch.cat((x_test,x_test[246:248,::]))
        y_test_bs= torch.cat((y_test.view(248),y_test[246:248].view(2))) 
    else:
        x_test_bs=x_test
        y_test_bs=y_test
    # Dataset class
    train_dataset= TensorDataset(x_train, y_train)
    test_dataset= TensorDataset(x_test_bs, y_test_bs)
    # DataLoaders 
    train_dl = DataLoader(train_dataset,batch_size=batch_size, shuffle=shuffle_stats,drop_last=drop_last)
    test_dl = DataLoader(test_dataset,batch_size=10, shuffle=False,drop_last=False)
    return train_dl, test_dl


def scoring_func(error_arr): 
    pos_error_arr = error_arr[error_arr >= 0] 
    neg_error_arr = error_arr[error_arr < 0]
    score = 0 
    for error in neg_error_arr:
            score = math.exp(-(error / 13)) - 1 + score 
    for error in pos_error_arr: 
            score = math.exp(error / 10) - 1 + score
    return score

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
