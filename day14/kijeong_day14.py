### transformer model을 이용하여 수자원 데이터 적합 ### 

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import tqdm
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from copy import deepcopy
import math

# Load data

data = pd.read_excel('C:\\Users\\kijeong\\research\\data\\수자원.xlsx')
data = data.drop(columns=['금곡1','금곡2','금곡3','회죽1','회죽2','회죽3'])
data.head()
data.shape[0] # 4321

plt.figure(figsize=(20,5))
plt.plot(range(len(data)), data['저수율'])
plt.plot(range(len(data)), data['저수량'])

# Data Preprocessing

data['dayofweek'] = data['날짜'].dt.dayofweek
data['month'] = data['날짜'].dt.month
data['year'] = data['날짜'].dt.year
data['day'] = data['날짜'].dt.day

def sin_transform(values):
    return np.sin(2*np.pi*values/len(set(values)))

def cos_transform(values):
    return np.cos(2*np.pi*values/len(set(values)))

data['dayofweek_sin'] = sin_transform(data['dayofweek'])
data['dayofweek_cos'] = cos_transform(data['dayofweek'])
data['month_sin'] = sin_transform(data['month'])
data['month_cos'] = cos_transform(data['month'])
data['day_sin'] = sin_transform(data['day'])
data['day_cos'] = cos_transform(data['day'])

data = data.set_index(['날짜'])
data
data.index = pd.to_datetime(data.index)
if not data.index.is_monotonic:
    data = data.sort_index()

data.drop(columns=['dayofweek', 'month', 'year', 'day'], inplace=True)
data.columns

scaler = MinMaxScaler()

train = data[:-14]
train.iloc[:, :-6] = scaler.fit_transform(train.iloc[:, :-6])
train.shape # 3956,22
train

test = data[-14:]
test.iloc[:, :-6] = scaler.transform(test.iloc[:, :-6])
test.shape # 365,22
test

'''
#### 실험 ####
train = train.iloc[:,[0]]
test = test.iloc[:, [0]]
'''

# Sliding Window Dataset

class windowDataset(Dataset):
    def __init__(self, y, input_window, output_window, num_features, stride=1):
        
        L = y.shape[0]
        
        num_samples = (L - input_window - output_window) // stride + 1
        # ((3956 - 365*2 - 365) // 1) + 1 = 2862

        X = np.zeros([input_window, num_samples, num_features])
        Y = np.zeros([output_window, num_samples, num_features]) 

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.transpose((1,0,2))
        Y = Y.transpose((1,0,2))
        self.x = X
        self.y = Y
        
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len

iw = 14*2 # ow의 2배로 설정, 730
ow = 14 # 1년의 데이터 예측, 365
#iw = 365*2 # ow의 2배로 설정, 730
#ow = 365 # 1년의 데이터 예측, 365

train_dataset = windowDataset(train, input_window=iw, output_window=ow, num_features = train.shape[1], stride=1)
train_loader = DataLoader(train_dataset, batch_size=64)

for b in train_loader:
    l, r = b
    break

# Modeling

# Embedding

l.shape
r.shape

class Linear1(nn.Module):
    def __init__(self, input_dim: int=22, d_model: int = 32):
        super(Linear1, self).__init__()
        self.d_model = d_model
        self.input_dim = input_dim

        self.linear = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model//2),
            nn.ReLU(),
            nn.Linear(self.d_model//2, d_model)
        )

    def forward(self, x):
        return self.linear(x.float())

linear1 = Linear1()
embedding = linear1(l)
embedding.shape


class Linear2(nn.Module):
    def __init__(self, input_dim: int=22, d_model: int = 32):
        super(Linear2, self).__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        self.linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            nn.ReLU(),
            nn.Linear(self.d_model//2, input_dim)
        )

    def forward(self, x):
        return self.linear(x.float())

class Linear3(nn.Module):
    def __init__(self, iw: int = 14*2, ow: int = 14):
        super(Linear3, self).__init__()
        self.iw = iw
        self.ow = ow
        
        self.linear = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        )

    def forward(self, x):
        return self.linear(x.float())



# Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 32, dropout: float = .1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout) 

        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.Tensor([10000.0])) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) 

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

pos = PositionalEncoding()
position = pos(embedding)
position.shape

# Attention

class Attention:
    def __init__(self, dropout: float = 0.):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1) 
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        #if mask is not None:
        #    scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = self.dropout(self.softmax(scores)) 

        return torch.matmul(p_attn, value)
    
    def __call__(self, query, key, value, mask=None):
        return self.forward(query, key, value, mask)


# Multi-Head Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, h: int = 8, d_model: int = 32, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // h 
        self.h = h 
        self.attn = Attention(dropout)
        self.lindim = (d_model, d_model)
        self.linears = nn.ModuleList([deepcopy(nn.Linear(*self.lindim)) for _ in range(4)])
 
        self.final_linear = nn.Linear(*self.lindim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        #if mask is not None:
        #    mask = mask.unsqueeze(1)
        
        query, key, value = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2) \
                             for l, x in zip(self.linears, (query, key, value))]

        nbatches = query.size(0)
        x = self.attn(query, key, value, mask=mask)
        
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.final_linear(x)


# Residuals and Layer Normalization

# Normalization

class LayerNorm(nn.Module):
    def __init__(self, features: int=22, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) 
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Residual Connection

class ResidualConnection(nn.Module):
    def __init__(self, size: int = 32, dropout: float = .1):
        super(ResidualConnection,  self).__init__()
        self.norm = LayerNorm(size) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Feed Forward

class FeedForward(nn.Module):
    def __init__(self, d_model: int = 32, d_ff: int = 2048, dropout: float = .1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.dropout(self.relu(self.l1(x))))


# Encoder

class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float = .1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn # Multi-Head Attention (Self-attention)
        self.feed_forward = feed_forward
        self.sub1 = ResidualConnection(size, dropout)
        self.sub2 = ResidualConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        x = self.sub1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.sub2(x, self.feed_forward)
        
class Encoder(nn.Module):
    def __init__(self, layer, n: int = 6):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n)])
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz,sz)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

src_mask = generate_square_subsequent_mask(l.shape[1])
encoder = Encoder(EncoderLayer(32, MultiHeadAttention(), FeedForward()))
encoder_output = encoder(position, src_mask)
encoder_output.shape

linear2 = Linear2()
lin2 = linear2(encoder_output)
lin2.shape
lin2 = lin2.transpose(1,2)

linear3 = Linear3()
lin3 = linear3(lin2).transpose(1,2)
lin3.shape


# Encoder-Decoder

class TF(nn.Module):
    def __init__(self, encoder: Encoder, linear1: Linear1, linear2: Linear2, linear3: Linear3,
                    pos_embed: PositionalEncoding):
        super(TF, self).__init__()

        self.encoder = encoder
        self.linear1 = linear1
        self.linear2 = linear2
        self.linear3 = linear3
        self.pos_embed = pos_embed
        
    def forward(self, src, src_mask):
        
        src = self.linear1(src)
        src = self.pos_embed(src)

        output = self.encoder(src, src_mask)
        output = self.linear2(output).transpose(1,2)
        output = self.linear3(output).transpose(1,2)

        return output
    

def make_model():
    encoder = Encoder(EncoderLayer(d, MultiHeadAttention(), FeedForward()))
    linear1 = Linear1()
    linear2 = Linear2()
    linear3 = Linear3()
    pos_embed = PositionalEncoding()
    model = TF(encoder, linear1, linear2, linear3, pos_embed)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

d = 32
lr = 0.001
model = make_model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epoch = 1


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz,sz)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask


# result = model(l[0].float(), r[0].float(), None, generate_square_subsequent_mask(r[0].shape[1]))


for b in train_loader:
    inputs, outputs = b
    break


model.train()
with tqdm(range(epoch)) as tq:
    for i in tq:
        total_loss = 0.0

        for (inputs, outputs) in train_loader:
            optimizer.zero_grad()
            src_mask = generate_square_subsequent_mask(inputs.shape[1])
            result = model(inputs.float(), src_mask)
            loss = criterion(result, outputs.float())
            loss.backward()
            optimizer.step()
            total_loss += loss

        tq.set_description("loss: {:0.6f}".format(total_loss.item() / len(train_loader)))



# Evaluate

model = torch.load('transformer.pth') # 서버에서 적합시킨 결과가 transformer.pth에 들어있음

def predict(inputs):
    model.eval()
    
    src_mask = generate_square_subsequent_mask(inputs.shape[1])
    
    result = model(inputs, src_mask)

    return result


pred = torch.tensor(train.iloc[-28:].values).float().unsqueeze(0)
out = torch.tensor(test.iloc[:14,:].values).unsqueeze(0)
prediction = predict(pred).squeeze(0)[:, :-6].detach().numpy()
real = test.iloc[:14, :-6].to_numpy()

prediction = scaler.inverse_transform(prediction)
real = scaler.inverse_transform(real)

prediction = prediction[: ,0]
real = real[: ,0]



dates = pd.date_range('2019-09-01','2019-09-14')
#dates = dates.drop('2020-06-30')

final = pd.DataFrame({'predict' : prediction, 'real' : real})
final.index = dates
final

prediction.min()
prediction.max()
real.min()
real.max()

# Visualization

plt.figure(figsize=(10,5))
plt.plot(final['real'], label="real")
plt.plot(final['predict'], label="predict")

plt.title("Prediction")
plt.legend()
plt.show()

def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPEval(prediction, real)