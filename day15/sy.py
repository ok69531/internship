# 수자원 test data에서 급격히 감소하는 부분 예측
# source code from https://doheon.github.io/코드구현/time-series/ci-4.transformer-post/

import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


df = pd.read_excel('../수자원.xlsx')
# df = pd.read_csv('../금융.csv')
# target = df['ETC-USD'].to_numpy()
feature = df.drop(['날짜', '저수량'], axis = 1).to_numpy()
target = df.저수율.to_numpy()

iw = 365
ow = 120

scaler = MinMaxScaler() 
feature = scaler.fit_transform(feature)
target = feature[:, 0]

num_test = -ow
x_train = feature[:num_test]
x_test = feature[num_test:]
y_train = target[:num_test]
y_test = target[num_test:]


class windowDataset(Dataset):
    def __init__(self, x, y, input_window=80, output_window=20, stride=5):
        #총 데이터의 개수
        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output
        # X = np.zeros([input_window, num_samples])
        X = np.zeros([input_window, x.shape[1], num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:, :, i] = feature[start_x:end_x, :]
            # X[:,i] = y[start_x:end_x]
            
            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.transpose((2, 0, 1))
        # X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = torch.tensor(X).to(torch.float32).to(device)
        self.y = torch.tensor(Y).to(torch.float32).to(device)
        
        self.len = len(X)
        
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.len


class TFModel(nn.Module):
    def __init__(self, num_feat, iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(num_feat, d_model//2),
            nn.LeakyReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LeakyReLU(),
            nn.Linear(d_model//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.LeakyReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        # output = self.transformer_encoder(src, srcmask)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).to(torch.float32) * (-math.log(10000.0) / d_model))
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


# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()       
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         #pe.requires_grad = False
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]
          

# class TransAm(nn.Module):
#     def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
#         super(TransAm, self).__init__()
#         self.model_type = 'Transformer'
        
#         self.src_mask = None
#         self.pos_encoder = PositionalEncoding(feature_size)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
#         self.decoder = nn.Linear(feature_size,1)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1    
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

#     def forward(self,src):
#         if self.src_mask is None or self.src_mask.size(0) != len(src):
#             device = src.device
#             mask = self._generate_square_subsequent_mask(len(src)).to(device)
#             self.src_mask = mask

#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
#         output = self.decoder(output)
#         return output

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask



lr = 1e-3
num_feat = x_train.shape[1]
d_model = 64
nhead = 2
# nhid = 64
nlayers = 2
dropout = 0.1


model = TFModel(num_feat, iw, ow, d_model, nhead, nlayers, dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_dataset = windowDataset(x_train, y_train, input_window=iw, output_window=ow, stride=3)
train_loader = DataLoader(train_dataset, batch_size=64)

# for b in train_loader:
#     inputs, outputs = b
#     break
# inputs.size()
# outputs.size()

epoch = 100
model.train()
progress = tqdm(range(epoch))
for i in progress:
    batchloss = 0.0
    for (inputs, outputs) in train_loader:
        optimizer.zero_grad()
        src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
        result = model(inputs, src_mask)
        loss = criterion(result, outputs[:,:,0].float().to(device))
        loss.backward()
        optimizer.step()
        batchloss += loss
    progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))
# torch.save(model.state_dict(), 'model.pth')


def evaluate():
    model.eval()
    
    inputs = torch.tensor(x_train[-iw:]).reshape(1,-1,num_feat).to(torch.float32).to(device)
    src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
    pred = model(inputs, src_mask)
    
    return pred.detach().cpu().numpy()


def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values


result = evaluate()
result = invTransform(scaler, result.squeeze(), '저수율', df.columns[2:])


plt.figure(figsize=(10,5))
plt.plot(df.저수율[num_test:].values, label="real")
plt.plot(result, label="pred")
plt.legend()
plt.show()
