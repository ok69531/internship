import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from tqdm import trange
import random
import math

try:
    import wandb
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable,"-m","pip","install","wandb"])
import wandb


wandb.init(project="internship")
wandb.run.name = 'transformer_ep1000'
# data import

data = pd.read_excel("수자원.xlsx", sheet_name=0)
plt.figure(figsize=(20,5))
plt.plot(range(len(data)), data["저수량"])

# data pre-precessing

data = data[["날짜","저수량"]]
data["날짜"] = pd.to_datetime(data["날짜"])
train = data[:-90].set_index(['날짜'])
test = data[-90:].set_index(['날짜'])

train.iloc[0]
train.shape
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

train["저수량"] = min_max_scaler.fit_transform(train["저수량"].to_numpy().reshape(-1,1))
test["저수량"] = min_max_scaler.transform(test["저수량"].to_numpy().reshape(-1,1))


class windowDataset(Dataset):
    def __init__(self, y, input_window, output_window, num_features ,stride = 1):
        L = y.shape[0]
        num_samples = (L - input_window - output_window) // stride + 1

        X = np.zeros([input_window, num_samples, num_features])
        Y = np.zeros([output_window, num_samples, num_features])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i,:] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i,:] = y[start_y:end_y]

        
        X = X.transpose((1,0,2))
        Y = Y.transpose((1,0,2))

        self.x = X
        self.y = Y

        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len


iw = 180
ow = 90
# iw = 730
# ow = 365

train_dataset = windowDataset(train, input_window=iw, output_window=ow,num_features=train.shape[1] ,stride=1)
train_loader = DataLoader(train_dataset, batch_size=64)


class TFModel(nn.Module):
    def __init__(self,iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask

best_valid =  float('inf')
lr = 1e-4
model = TFModel(iw, ow, 64, 4, 2, 0.1)
# model = TFModel(730, 365, 512, 8, 4, 0.1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epoch = 100
model.train()
# progress = tqdm(range(epoch))
for i in range(epoch+2):
    total_loss = 0.0
    for (inputs, outputs) in train_loader:
        optimizer.zero_grad()
        src_mask = model.generate_square_subsequent_mask(inputs.shape[1])
        result = model(inputs.float(),  src_mask)
        loss = criterion(result, outputs[:,:,0].float())
        loss.backward()
        optimizer.step()
        total_loss += loss
    # progress.set_postfix("loss: {:0.6f}".format(total_loss.item() / len(train_loader)))
    print(f'epoch: {i}, loss:{total_loss/len(train_loader)}')
    wandb.log({'epoch': i, 'loss': total_loss/len(train_loader)})
    if loss<=best_valid:
        torch.save(model, 'best_ep1.pth')
        patient=0
        best_valid=loss
    else:
        patient+=1
        if patient>=10:
            break    

# wandb.finsh()