# Load data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import trange
import random

data = pd.read_csv(r"C:\winter\my_practice\day10\서인천IC-부평IC 평균속도.csv", encoding='CP949')
plt.figure(figsize=(20,5))
plt.plot(range(len(data)), data["평균속도"])
data.head()
#Data Preprocessing
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

#fit_transform은 fit과 transform을 한 번에 합쳐놓은 함수(train_set에서 활용)
#아무래도 이사람은 순서를 잘못한듯 하다
#fit : 변환 계수 추정, transfor : 실제로 자료를 변환
# .fit_transform은 2d array를 input으로 받기 때문에 reshape로 차원 맞춰줌
data["평균속도"] = min_max_scaler.fit_transform(data["평균속도"].to_numpy().reshape(-1,1))


#마지막 일주일을 예측하기 위해 기준을 잡고 train과 test로 분류
train = data[:-24*7]
train = train["평균속도"].to_numpy()


test = data[-24*7:]
test = test["평균속도"].to_numpy()
#sliding window dataset
from torch.utils.data import DataLoader, Dataset

class windowDataset(Dataset):
    def __init__(self, y, input_window, output_window, stride=1):
        #총 데이터의 개수
        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        #식이 헷갈릴수 있는데 몫 구하고 1을 더하는거임!!
        #첫 윈도우가 빼기로 빠졌으니까 1 더해주기
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output : shape = (window 크기, sample 개수)
        # 미리 size에 맞게 zeros로 생성
        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        #실수에 for문 돌리려구 arange 사용
        for i in np.arange(num_samples):
            start_x = stride*i #i번째 window의 시작 index를 곱으로 표현
            end_x = start_x + input_window #window 크기만큼 더하기
            X[:,i] = y[start_x:end_x] #X의 i번째 윈도우를 X에 열로 저장

            #X와 마찬가지로 output에 대해 같은 순서를 진행
            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        # reshape를 통해 2d array인 X와 Y에 3차원을 1로 추가
        # transpose로 window와 num_sample의 자리를 바꿔줌
        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = X
        self.y = Y
        
        self.len = len(X)
    # __getitem__은 class로 생성된 객체의 인덱스에 접근할 때
    # 호출되는 메서드이다.
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len

iw = 24*14
ow = 24*7

train_dataset = windowDataset(train, input_window=iw, output_window=ow, stride=1)

#window의 개수를 batch로 나눠주는데 솔직히 여기서 73개를 왜 size 64짜리로 나눴는지 모르겟음
train_loader = DataLoader(train_dataset, batch_size=64)


# train_dataset.x.shape = (73, 336, 1)
# train_loader의 첫 x = (64, 336, 1)
# 두번째는 (9, 336, 1)

# Modeling
# torch.nn.Module을 활용한 encoder decoder model
# encoder : input을 통해 decoder에 전달할 hidden state 생성
# decoder : input의 마지막 값과 encoder에서 받은 hidden state를 이용하여 한 개의 값을 예측
# encoder decoder : 위의 두 모델을 합쳐줌.
# 원하는 길이의 아웃풋이 나올 때까지 decoder를 여러 번 실행시켜서 최종 output 생성
# 원활한 학습을 위해 디코더의 인풋으로 실제 값을 넣는 teach forcing을 구현.

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



# Lstm decoder
# sequence의 이전 값 하나와, 이전 결과의 hidden state를 입력 받아서 다음 값 하나를 예측
# 마지막에 fc layer를 연결해서 input size와 동일하게 크기를 맞춰준다.

class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #batch_first는 batch_size가 size의 제일 앞에 오게 해주는 옵션(우리가 원래 알던대로)
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        # 왜 unsqueeze 해야하는지 이해하자!
        # nn.LSTM의 input은 (N,L,H_in), output은 (N,L, D*H_out)이다.
        # N:batch_size, L:sequence_length, D: direction 옵션 켜져있으면 2또는 1, 평소엔 1
        # x_input이 추후 encoder를 통과한 (64*1)이므로 unsqueeze(-1)로 3차원으로 다시 돌려줌
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(-1), encoder_hidden_states)
        # 이때 lstm의 공식문서에선 output, (hidden, cell_state)를 받는다고 되어있는데 왜 두 개로 받을까???
        # hidden자리에 하나만 넣어도 (hidden, cell_state)로 받아주니 당황하지 말자(편의성을 위해서인듯?)
        # 3개 변수로 받으면 hidden, cell 따로 받아준다
        output = self.linear(lstm_out)

        return output, self.hidden


# encoder decoder
# 위의 두 모델을 합쳐준다.
# 인코더를 한 번 실행시키고 인코더에서 전달받은 hidden state와 input의 마지막 값을 디코더에 전달해 다음 예측값을 구함
# 여기서 나온 값과 hidden state를 반복적으로 사용해서 원하는 길이가 될때까지 decoder를 실행한다.
# decoder의 인풋으로 이전 예측값이 아닌 실제 값을 사용하는 teacher forcing도 구현

class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size #1
        self.hidden_size = hidden_size #16

        self.encoder = lstm_encoder(input_size= input_size, hidden_size= hidden_size)
        self.decoder = lstm_decoder(input_size= input_size, hidden_size= hidden_size)
    
    def forward(self, inputs, targets, target_len, teacher_forching_ratio):
        batch_size = inputs.shape[0] # 1번이라 치면 (64,336,1)에서 64
        input_size = inputs.shape[2] # (64,336,1)에서 1

        outputs = torch.zeros(batch_size, target_len, input_size)

        _,hidden = self.encoder(inputs) #encoder 모델에서 decoder를 돌릴 때 필요한 hidden만 가져옴
        decoder_input = inputs[:, -1, :] # encoder의 마지막 input을 decoder의 첫 번째 input으로
        #여기서 한 줄을 가져오기 때문에 차원이 줄어든다.

        # 원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1) #1인 차원을 모두 제거, output의 한 열로 넣기 위해 squeeze(1)

            # teacher forcing을 구현한다.
            # 0~1 사이의 값을 랜덤으로 뽑아 teacher forcing보다 작으면 다음 input올 예측값이 아닌 실제값 사용
            if random.random() < teacher_forching_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
            outputs[:, t, :] = out
        
        return outputs

    #편의성을 위해 예측해주는 함수도 생성
    def predict(self, inputs, target_len):
        self.eval() #예측이니까 평가 때처럼 .eval 해줌
        inputs = inputs.unsqueeze(0) # predict는 model.predict(input, target_len)과 같이 사용할거다
        # 이 때 loader가 아닌 train dataset을 짤라서 이용하므로 3차원 맞춰주고 배치를 1로 하려고 unsqueeze(0)
        # 336 * 1 이 들어와 1*336 *1이 된다
        batch_size = inputs.shape[0] # 1
        input_size = inputs.shape[2] # 1
        outputs = torch.zeros(batch_size, target_len, input_size) #(1, 168, 1)
        # 이후 예측이기 때문에 trainc forcing이 없이 예측값만 저장할 수 있도록 encoder, decoder 활용
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:,-1,:]
        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1)
            decoder_input = out
            outputs[:, t, :] = out
        return outputs.detach().numpy()[0,:,0] #detach는 필요없는 gradient 어쩌고 떼고 tensor의 값만 줌


# Train
import torch.optim as optim

model = lstm_encoder_decoder(input_size=1, hidden_size=16)

learning_rate=0.01
epoch = 1000
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

from tqdm import tqdm

model.train()
# 진행상황 확인하기 위해 tqdm 넣었음
with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x,y in train_loader:
            optimizer.zero_grad()
            x = x.float()
            y = y.float()
            output = model(x, y, ow, 0.6) # ow는 앞에서 설정한 out window 크기
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # 배치 하나 끝나면 loss를 tqdm의 set_postfix를 이용하려 로깅
        # 1 epoch마다 loss를 진행도 옆에 보여줌
        # tqdm의 print와 비슷한 역할
        tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_loader)))

# evaluate
# model.predict의 input 차원에 맞추기 위해 train data를 tesnor로 바꾼 뒤
# reshape를 통해 336*1로 맞춰주고 예측함
predict = model.predict(torch.tensor(train[-24*7*2:]).reshape(-1,1).float(), target_len=ow)
# 실제 값 역시 뽑아놓음
real = data["평균속도"].to_numpy()

# 이미 정규화를 해두었기에 inverse_transform을 통해 원래대로 돌려준다.
predict = min_max_scaler.inverse_transform(predict.reshape(-1,1))
real = min_max_scaler.inverse_transform(real.reshape(-1,1))

plt.figure(figsize=(20,5))
plt.plot(range(400,744), real[400:], label="real") #너무 넓어지니까 400부터만 실제값 가져옴
plt.plot(range(744-24*7,744), predict[-24*7:], label="predict") # 마지막 168개에 대한 예측값의 그래프 겹쳐그림

plt.title("Test Set")
plt.legend()
plt.show()

# 별거 없고 그냥 공식을 실제로 구현한거
# 예측과 실제의 차이의 절대값을 실제로 나눠서 평균냄
# 낮을수록 좋음!
def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
MAPEval(predict[-24*7:],real[-24*7:])

