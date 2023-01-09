import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from tqdm import trange
import random

# Load Data

data = pd.read_csv('data/서인천IC-부평IC 평균속도.csv', encoding='CP949')
# 744 rows X 3 columns
plt.figure(figsize=(20,5))
plt.plot(range(len(data)), data['평균속도'])
data.head()


# Data Preprocessing

from sklearn.preprocessing import MinMaxScaler # 정규화
# MinMaxScaler는 Max=1, Min=0으로 조정해주는 스케일링, 단 아웃라이어에 취약함
min_max_scaler = MinMaxScaler() # 클래스 객체 생성

data['평균속도'].to_numpy().shape # (744,)
data['평균속도'].to_numpy().reshape(-1,1).shape # (744, 1)

data['평균속도'] = min_max_scaler.fit_transform(data['평균속도'].to_numpy().reshape(-1,1))
data['평균속도']
# fit_transform은 fit과 transform을 한 번에 처리할 수 있게 하는 메소드로 test data에는 사용하면 안 됨
# fit은 데이터를 학습시키는 메소드, transform은 실제로 학습시킨 것을 적용하는 메소드
# 만약 test data에도 fit을 해버리면 scaler가 기존에 train data에 fit한 기준을 다 무시하고 test data에 새로운 mean, var값을 얻으면서 test data까지 학습해버림
# 따라서 절대로 test data에는 fit이나 fit_transform 메소드를 사용하면 안 됨

train = data[:-24*7]
train = train["평균속도"].to_numpy()
train.shape[0] # 576

test = data[-24*7:]
test = test["평균속도"].to_numpy()
test.shape[0] # 168
# 마지막 일주일의 데이터를 예측하는 것이 목표이므로 train, test set을 마지막 일주일을 기준으로 나눔


# Sliding Window Dataset

# 학습을 위해서는 input data와 output data가 필요함
# 시계열 예측을 위해 데이터의 일정한 길이의 input window, output window를 설정하고, 데이터의 처음 부분부터 끝까지 sliding시켜서 dataset 생성

from torch.utils.data import DataLoader, Dataset

class windowDataset(Dataset):
    def __init__(self, y, input_window, output_window, stride=1):
        
        L = y.shape[0] # 총 데이터의 개수(row의 수)
        
        # stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1
        # ((576 - 24*14 - 24*7) // 1) + 1 = 73

        # input과 output : shape = (window 크기, sample 개수)
        X = np.zeros([input_window, num_samples]) # input
        Y = np.zeros([output_window, num_samples]) # output

        for i in np.arange(num_samples):
            start_x = stride*i # stride가 1이므로 1,2,3,...
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x] # 데이터 한 개의 사이즈가 input_window의 사이즈와 동일, 총 num_samples개만큼 들어감

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y] # 데이터 한 개의 사이즈가 output_window의 사이즈와 동일, 총 num_samples개만큼 들어감

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2)) # 기존 순서 0,1,2를 1,0,2로 바꾸라는 것, (73,336,1)
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2)) # (73,168,1)
        self.x = X
        self.y = Y
        
        self.len = len(X)
    def __getitem__(self, i): # 클래스의 인덱스에 접근할 때 자동으로 호출되는 함수
        # 이걸 해두면 인덱스 접근이 가능해져 인덱스를 통해 원하는 값을 얻을 수 있음
        return self.x[i], self.y[i]
    def __len__(self):
        # len()을 이용할 수 있게 해주는 함수
        return self.len

# 위 클래스에서 헷갈리는 부분 펼쳐서 확인
x = np.zeros([336, 73])
y = np.zeros([168, 73])

for i in np.arange(73):
    start_x = i
    end_x = start_x + 336
    x[:,i] = train[start_x:end_x]

    start_y = i + 336
    end_y = start_y + 168
    y[:,i] = train[start_y:end_y]
x.shape # 336,73
y.shape # 168.73

x.reshape(x.shape[0], x.shape[1], 1).transpose((1,0,2)).shape
y.reshape(y.shape[0], x.shape[1], 1).transpose((1,0,2)).shape
# 여기까지


iw = 24*14 # ow의 2배로 설정, 336
ow = 24*7 # 일주일 간의 데이터를 예측해야 하므로 24*7 = 168

train_dataset = windowDataset(train, input_window=iw, output_window=ow, stride=1)
# windowDataset클래스의 객체 train_dataset 생성
# 총 73개의 시퀀스, 시퀀스 한 개의 길이는 input 336, output 168 
train_loader = DataLoader(train_dataset, batch_size=64)
# DataLoader클래스의 객체 train_loader 생성
# 64개로 나누므로 첫번째 loader의 길이는 64, 그 다음은 73-64=9가 됨


# Modeling
# encoder : input을 통해 decoder에 전달할 hidden state 생성
# decoder : input의 마지막 값과 encoder에서 받은 hidden state를 이용하여 한 개의 값을 예측
# encoder decoder : 두 모델을 합치고, 원하는 길이의 output이 나올때까지 decoder를 여러번 실행시켜서 최종 output을 생성
# 원활한 학습을 위해 decoder의 input으로 실제 값을 넣는 teach forcing을 구현
import torch
import torch.nn as nn

# encoder : input으로부터 입력을 받고 lstm을 이용하여 decoder에 전달할 hidden state 생성
class lstm_encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)
        # batch_first=True면 입력 및 출력 텐서가 (sequence, batch, feature) -> (batch, sequence, feature)순으로 변함
        # 즉 batch단위로 학습을 진행하게 하기 위해 사용

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden


# decoder : sequence의 이전값 하나와, 이전 결과의 hidden state를 입력 받아서 다음 값 하나를 예측
# 마지막에 fc layer(linear)를 연결해서 input size와 동일하게 크기를 맞춰줌
class lstm_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size) # 최종값을 반환하기 위해서 Linear 적용

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(-1), encoder_hidden_states)
        # lstm_out.shape=(64,336,16), hidden.shape=(2,1,64,16)
        # hidden에는 cell state와 hidden state가 모두 포함되어 있음 -> 4차원
        # hidden state는 (num_layers, batch_size, hidden_size) = (1,64,16)임, cell state도 마찬가지임
        # encoder를 다 거쳐간 후 input의 마지막 값 하나만 뽑기 때문에 차원이 하나 줄어듦
        # 따라서 이를 맞춰주기 위해서 unsqueeze(-1)을 사용함
        # (64,16) -> (64,16,1)
        # lstm을 거치면 (64,16,16)으로 shape가 변함
        output = self.linear(lstm_out)
        # (64,16,16) -> (64,16,1)

        return output, self.hidden

l =[]
r =[]
for x,y in train_loader:
    x = x.float()
    l.append(x)
    y = y.float()
    r.append(y)
r[0].shape
l[0].shape

lstm = nn.LSTM(input_size = 1, hidden_size = 16, num_layers = 1, batch_first=True)
out, hidden = lstm(l[0])
len(hidden[0][0][0]) # (2,1,64,16), hidden state와 cell state 반환
linear = nn.Linear(16, 1)
linear(out).squeeze(1).shape
a = out[:,-1,:]
a.shape
b = a.unsqueeze(1)
b.shape
c = a.unsqueeze(-1)
c.shape
b.squeeze(1).shape
c.squeeze().shape
out, hidden = lstm(b)
out.shape
linear(out).shape
linear(out).transpose((0,2,1))

torch.tensor([[[3],[3],[1]]]).squeeze(0)

# encoder_decoder : 위의 두 모델을 합쳐줌
# encoder를 한 번 실행시키고 encoder에서 전달받은 hidden state와 input의 마지막 값을 decoder에 전달해서 다음 예측값을 구함
# 이 때 나온 값과 hidden state를 반복적으로 사용해서 원하는 길이가 될 때까지 decoder 실행
# ex. 번역의 경우 문장에 포함된 단어가 10개면 10번까지 실행한다는 의미
# decoder의 input으로 이전 예측값이 아닌 실제 값을 사용하는 teacher forcing도 구현
class lstm_encoder_decoder(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio):
        batch_size = inputs.shape[0] # 64
        input_size = inputs.shape[2] # 1

        outputs = torch.zeros(batch_size, target_len, input_size) # (64,168,1)

        _, hidden = self.encoder(inputs)
        
        decoder_input = inputs[:, -1, :]

        # 원하는 길이(target_len)가 될 때까지 decoder 실행
        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1)

            # teacher forcing 구현
            # teacher forcing이란 target word를 decoder의 다음 입력으로 넣어주는 것을 말함
            # 만약 t시점의 예측이 잘못되었다면 t+1시점부터는 다 잘못되게 된다
            # 따라서 이를 방지하고자 사용함
            # 학습이 빠르다는 장점이 있지만, 노출 편향 문제가 존재함
            # teacher forcing에 해당하면 다음 input값으로는 예측한 값이 아니라 실제 값을 사용
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
                # 해당 시점의 targets값을 input으로 사용함

            else:
                decoder_input = out

            outputs[:,t,:] = out
        
        return outputs

    # 편의성을 위해 예측해주는 함수도 생성
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

            return outputs.detach().numpy()[0,:,0]


# Train
# 생성한 모델과 데이터를 사용하여 훈련
import torch.optim as optim

model = lstm_encoder_decoder(input_size=1, hidden_size=16)
learning_rate=0.01
epoch = 1000
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

from tqdm import tqdm

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
        # set_postfix : tqdm에서의 print


# Evaluate
# 학습된 모델을 사용해서 훈련집합에는 포함되지 않았던 마지막 일주일의 데이터 예측
predict = model.predict(torch.tensor(train[-24*7*2:]).reshape(-1,1).float(), target_len=ow)
real = data['평균속도'].to_numpy()

predict = min_max_scaler.inverse_transform(predict.reshape(-1,1))
real = min_max_scaler.inverse_transform(real.reshape(-1,1))
# inverse_transform은 scaling 한 것을 원래대로 돌리는 함수


# 시각화
plt.figure(figsize=(20,5))
plt.plot(range(400,744), real[400:], label="real")
plt.plot(range(744-24*7,744), predict[-24*7:], label="predict")

plt.title("Test Set")
plt.legend()
plt.show()


# Loss 계산
# MAPE 사용 (Mean Absolute Percentage Error)
def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPEval(predict[-24*7:],real[-24*7:]) # 9.32220190921406