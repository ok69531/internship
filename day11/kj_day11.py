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

train = data[:-365]
train.iloc[:, :-6] = scaler.fit_transform(train.iloc[:, :-6])
train.shape # 3956,22
train

test = data[-365:]
test.iloc[:, :-6] = scaler.transform(test.iloc[:, :-6])
test.shape # 365,22
test

'''
X = np.zeros([730, 2862, 22]) # input
Y = np.zeros([365, 2862, 22]) # output

for i in np.arange(2862):
    start_x = i # stride가 1이므로 0,1,2,3,...
    end_x = start_x + 730
    X[:,i] = train[start_x:end_x] # 데이터 한 개의 사이즈가 input_window의 사이즈와 동일, 총 num_samples개만큼 들어감

    start_y = i + 730
    end_y = start_y + 365
    Y[:,i] = train[start_y:end_y] # 데이터

X.transpose(1,0,2).shape
Y.transpose(1,0,2).shape
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

        X = X.transpose((1,0,2)) # 기존 순서 0,1,2를 1,0,2로 바꾸라는 것 (2862,730,22)
        Y = Y.transpose((1,0,2)) # (2862,365,22)
        self.x = X
        self.y = Y
        
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len

iw = 365*2 # ow의 2배로 설정, 730
ow = 365 # 1년의 데이터 예측, 365

train_dataset = windowDataset(train, input_window=iw, output_window=ow, num_features = train.shape[1], stride=1)
train_loader = DataLoader(train_dataset, batch_size=64)


'''
l =[]
r =[]
for x,y in train_loader:
    x = x.float()
    l.append(x)
    y = y.float()
    r.append(y)
r[0].shape
l[44].shape
l[0]
'''
# modeling

class lstm_encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)


    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden



class lstm_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        
        lstm_out, self.hidden = self.lstm(x_input.view((-1,1,22)), encoder_hidden_states)
        
        output = self.linear(lstm_out)

        return output, self.hidden

'''
lstm = nn.LSTM(input_size=22, hidden_size=16, num_layers=1, batch_first=True)
out, hidden = lstm(l[0])
out.shape # 64,730,16
hidden[0].shape # 1,64,16
l[0][:,-1,:].unsqueeze(-1).shape # 64,22,1

encoder = lstm_encoder(input_size=22, hidden_size=16, num_layers=1)
out_e, hidden_e = encoder(l[0])
out_e.shape # 64,730,16
hidden_e[0].shape # 1,64,16

decoder_input = l[0][:,-1,:]
decoder_input.shape # 64, 22
decoder_input.unsqueeze(-1).shape
linear = nn.Linear(1,22)
inp = linear(decoder_input.unsqueeze(-1))
decoder = lstm_decoder(input_size=22, hidden_size=16, num_layers=1)
out_d, hidden_d = decoder(inp, hidden_e)
out_d.shape # 64,22,22
linear2 = nn.Linear(22,1)
oo = linear2(out_d).view((64,1,22))
hidden_d[0].shape # 1,64,16

outputs = torch.zeros(64, 365, 22)
for t in range(365):
    out, hidden = decoder(decoder_input, hidden_e)
    out = out.squeeze(1)
    outputs[:,t,:] = out
outputs
'''

class lstm_encoder_decoder(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio):
        batch_size = inputs.shape[0] # 64
        input_size = inputs.shape[2] # 22

        outputs = torch.zeros(batch_size, target_len, input_size) # (64,365,22)

        _, hidden = self.encoder(inputs)
        
        decoder_input = inputs[:, -1, :] # shape : (64,22)

        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1) 

            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
                

            else:
                decoder_input = out

            outputs[:,t,:] = out
        
        return outputs

    def predict(self, inputs, target_len):
            self.eval()
            
            inputs = inputs.unsqueeze(0) # 위에서의 inputs과 다른 inputs

            batch_size = inputs.shape[0]
            input_size = inputs.shape[2]

            outputs = torch.zeros(batch_size, target_len, input_size)

            _, hidden = self.encoder(inputs)

            decoder_input = inputs[:, -1, :]

            for t in range(target_len): 
                out, hidden = self.decoder(decoder_input, hidden)
                out =  out.squeeze(1)
                decoder_input = out
                outputs[:,t,:] = out

            return outputs[0,:,:].detach().numpy()


# Train

model = lstm_encoder_decoder(input_size=22, hidden_size=16)
learning_rate=0.01
epoch = 1000
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()
best_valid = float('inf')
patient = 0

model.train()
with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x, y in train_loader:

            optimizer.zero_grad()
            
            x = x.float()
            y = y.float()

            output = model(x, y, ow, 0.6)

            loss = criterion(output, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        tr.set_postfix(loss = "{0: .5f}".format(total_loss/len(train_loader)))

        if loss <= best_valid:
            torch.save(model, 'best.pth') # pytorch에서는 모델을 저장할 때 .pt 또는 .pth 확장자를 사용
            patient = 0
            best_valid = loss
        else:
            patient += 1
            if patient >= 20:
                break


# Prediction

pred = torch.tensor(train.iloc[-730:].values).float()
predict = model.predict(pred, target_len=ow)[:, :-6]
real = test.iloc[:, :-6].to_numpy()

predict = scaler.inverse_transform(predict)
real = scaler.inverse_transform(real)

predict = predict[: ,0]
real = real[: ,0]


# Visualization

plt.figure(figsize=(20,5))
plt.plot(range(365), real, label="real")
plt.plot(range(365), predict, label="predict")

plt.title("Test Set")
plt.legend()
plt.show()

def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPEval(predict, real)