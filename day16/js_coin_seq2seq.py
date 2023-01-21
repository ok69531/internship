import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm import trange
import random


# data import

data = pd.read_csv("금융.csv")
plt.figure(figsize=(20,5))
plt.plot(range(len(data)), data["ETC-USD"])




data = data.iloc[:,0:6].drop(["BNB-USD","BTC-USD","DOGE-USD","EOS-USD"], axis=1)
data["Date"] = pd.to_datetime(data["Date"])

train = data[:-120].set_index(['Date'])
test = data[-120:].set_index(['Date'])

train.head()

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()


train= min_max_scaler.fit_transform(train)
test = min_max_scaler.transform(test)

test

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


iw = 365
ow = 120

train_dataset = windowDataset(train, input_window=iw, output_window=ow,num_features=train.shape[1] ,stride=1)
train_loader = DataLoader(train_dataset, batch_size=16)



import torch.nn as nn
# Lstm encoder
class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden



class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.view([-1,1,1]), encoder_hidden_states)

        output = self.linear(lstm_out)

        return output, self.hidden



class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size 
        self.hidden_size = hidden_size 

        self.encoder = lstm_encoder(input_size= input_size, hidden_size= hidden_size)
        self.decoder = lstm_decoder(input_size= input_size, hidden_size= hidden_size)
    
    def forward(self, inputs, targets, target_len, teacher_forching_ratio):
        batch_size = inputs.shape[0] 
        input_size = inputs.shape[2] 

        outputs = torch.zeros(batch_size, target_len, input_size)

        _,hidden = self.encoder(inputs) 
        decoder_input = inputs[:, -1, :] 


        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1)

            if random.random() < teacher_forching_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
            outputs[:, t, :] = out
        
        return outputs


    def predict(self, inputs, target_len):
        self.eval()
        inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0] 
        input_size = inputs.shape[2] 
        outputs = torch.zeros(batch_size, target_len, input_size) 
    
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:,-1,:]
        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1)
            decoder_input = out
            outputs[:, t, :] = out
        return outputs

# Train
import torch.optim as optim

model = lstm_encoder_decoder(input_size=1, hidden_size=64)

learning_rate=0.001
epoch = 50
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

patient=0

from tqdm import tqdm

model.train()


with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x,y in train_loader:
            optimizer.zero_grad()
            x = x.float()
            y = y.float()
            output = model(x, y, ow, 0.6) 
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_loader)))


model.eval()

pred_input = torch.tensor(train[:-365]).float()
predict = model.predict(pred_input, target_len=ow)[-1].detach().numpy()


real = test


predict = min_max_scaler.inverse_transform(predict)
real = min_max_scaler.inverse_transform(real)

real = pd.DataFrame(real).to_numpy()

dates = pd.date_range('2022-09-11','2023-01-08')
len(dates)



final = pd.DataFrame({'predict' : predict.squeeze(1), 'real' : real.squeeze(1)})
final.index = dates
final

predict.min()
predict.max()
real.min()
real.max()

#x_tick = np.array(list(range(365)))
#plt.figure(figsize=(20,5))
#plt.plot(x_tick, real, label="real") 
#plt.plot(x_tick, predict, label="predict")

#plt.title("Test Set")
#plt.legend()
#plt.show()

# Visualization

plt.figure(figsize=(10,5))
plt.plot(final['real'], label="real")
plt.plot(final['predict'], label="predict")


plt.title("Prediction")
plt.legend()
plt.show()



def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
MAPEval(predict,real)

