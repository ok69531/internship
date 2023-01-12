import plotly.graph_objs as go
from plotly.offline import iplot
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# 데이터 준비

# dataset and plot
def plot_dataset(df, title):
    data = []
    value = go.Scatter(
        x=df.index,
        y=df.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)

df = pd.read_csv('C:/Users/kijeong/research/archive/PJME_hourly.csv')
# pjme : 
df = df.set_index(['Datetime'])
df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()
    
df = df.rename(columns={'PJME_MW': 'value'})

#plot_dataset(df, title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')

# Using lagged observations as features
def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n
    
input_dim = 100

df_generated = generate_time_lags(df, input_dim)
df_generated
# 하나는 실제 값에 대한 것이고, 나머지는 각 행에서 이전 100개의 관측치에 대한 것임


# Generating features from timestamp
df_features = (
                df
                .assign(hour = df.index.hour)
                .assign(day = df.index.day)
                .assign(month = df.index.month)
                .assign(day_of_week = df.index.dayofweek)
                .assign(week_of_year = df.index.week)
              )

df_features
# One-Hot Encoding
def onehot_encode_pd(df, col_name):
    dummies = []
    for column in col_name:
        dummy = pd.get_dummies(df[column], prefix=column)
        dummies.append(dummy)
    cat_dummies = pd.concat(dummies, axis=1)
    return pd.concat([df, cat_dummies], axis=1)

df_features = onehot_encode_pd(df_features, ['month','day','day_of_week','week_of_year'])
df_features


# Generating cyclical time features
def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
             }
    return df.assign(**kwargs).drop(columns=[col_name])

df_features = generate_cyclical_features(df_features, 'hour', 24, 0)
df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)
df_features = generate_cyclical_features(df_features, 'month', 12, 1)
df_features = generate_cyclical_features(df_features, 'week_of_year', 52, 0)

df_features

# Other features
from datetime import date
import holidays

us_holidays = holidays.US()

def is_holiday(date):
    date = date.replace(hour = 0)
    return 1 if (date in us_holidays) else 0

def add_holiday_col(df, holidays):
    return df.assign(is_holiday = df.index.to_series().apply(is_holiday))


df_features = add_holiday_col(df_features, us_holidays)
df_features


# Data Split
# time-dependent data -> time sequences를 그대로 유지해야함 -> shuffle = False
from sklearn.model_selection import train_test_split

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, 'value', 0.2)
# value column의 값은 y에 넣고, target으로 삼으며, train, validation, test set을 0.6:0.2:0.2로 분리, 이 때 섞지 않고 순차적으로 분리
# value라는 column을 없애고, 나머지 부분을 X에 넣고, train, validation, test set을 0.6:0.2:0.2로 분리, 이 때 섞지 않고 순차적으로 분리
df_features.shape # 145366, 114
X_train.shape # 87219, 113
X_val.shape # 29073, 113
X_test.shape # 29074, 113
y_train.shape # 87219, 1
y_val.shape # 29073, 1
y_test.shape # 29074, 1
# 현재는 dataframe


# Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)
# 현재는 numpy array


# Loading the datasets into DataLoaders
# 데이터를 로드하고 mini-batch training을 수행할 수 있도록 배치로 분할
from torch.utils.data import TensorDataset, DataLoader

batch_size = 64

train_features = torch.Tensor(X_train_arr)
train_targets = torch.Tensor(y_train_arr)
val_features = torch.Tensor(X_val_arr)
val_targets = torch.Tensor(y_val_arr)
test_features = torch.Tensor(X_test_arr)
test_targets = torch.Tensor(y_test_arr)
# 현재 Tensor

train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(test_features, test_targets)
train[0]
# 이렇게 TensorDataset으로 결합하면 반복과 슬라이싱에 유용함
# 슬라이싱 할 경우 두 데이터를 tuple형식으로 반환

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
# dataloader는 batch에 대해서 반복하기 편리하게 해줌
# for문에 적용시 지정된 batch_size만큼 알아서 분리해서 하나씩 넣어줌
# shuffle=True로 하면 랜덤하게 뽑아서 주고, drop_last=False로 하면 마지막에 나머지 부분을 하나의 배치로 간주해서 넣어줌
# batch가 1인 test_loader_one의 경우 필수는 아니지만 있으면 좋음

df_features['value'].max() # 62009.0
df_features['value'].min() # 14544.0

# RNN 구축
import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
       
        # self.softmax = nn.Softmax(dim=1)
        # 분류가 아닌 예측이기 때문에 확률값을 출력해주는 softmax를 통과시키지 말아야 함
        # dim=1 : 두번째 차원에 대해서 LogSoftmax를 적용한다는 의미
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        # (64, 1, 113) + (64, 1, 128) -> (64, 1, 241)

        hidden = self.i2h(combined) # (64, 1, 241) -> (64, 1, 128)
        output = self.i2o(combined) # (64, 1, 241) -> (64, 1, 1)

        #output = self.softmax(output) # LogSoftmax 적용
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size) # 원래 hidden size만큼 영벡터 생성

n_input = X_train.shape[1] # 113
n_hidden = 128
n_output = y_train.shape[1] # 1

model = RNN(n_input, n_hidden, n_output)
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 네트워크 학습   

def train(model, criterion, loader):

    model.train()

    for step, (x_batch, y_batch) in enumerate(loader):

        hidden = model.initHidden()

        optimizer.zero_grad()

        for cell, target in zip(x_batch, y_batch):

            cell = cell.unsqueeze(0)

            output, hidden = model(cell, hidden)
        # 위에서 만든 RNN클래스는 RNN 전체를 실행시켜주는게 아니라 하나의 RNN Cell임
        # 그래서 RNN을 돌리려면 하나의 시퀀스를 설정해서 for문을 통해 위에서 만든 RNN Cell을 반복적으로 실행시켜야 함
        # 여기서는 loader를 통해 불러온 데이터를 위에서 설정한 batch_size로 나누어서 가져온 x_batch가 RNN 시퀀스가 되는거고
        # 시퀀스의 길이가 x_batch의 길이인 64가 된다고 보는게 맞음
        # x_batch에서 한 줄씩 가져와서 cell에 저장 후 차원을 맞춰주고, 위에서 초기화한 hidden을 가지고 와서 rnn에 넣어준 다음
        # 하나의 RNN 시퀀스를 다 돌리고 나서 최종적으로 나온 output과 y_batch의 마지막 데이터를 이용해서 loss를 계산함
        # 이 작업을 총 loader만큼 반복하면서 학습을 진행함

        loss = criterion(output.squeeze(0), target)
        # target에 y_batch에서 뽑은 마지막 값이 저장되어 있음
        # output과 target의 차원이 맞지 않으므로 squeeze(0)를 통해 차원을 1로 통일

        loss.backward()

        optimizer.step()
        # 잘 안 돌아가서 6일차에서 구현한 SGD 대신에 optimizer로 Adam을 사용하였습니다...

        # torch.autograd.set_detect_anomaly(True)
        # autograd 수행 시 이상현상이 발생하면 그 내용을 알려줌

    return loss


# 결과 평가
@torch.no_grad()
def evaluate(model, criterion, loader):

    model.eval() 
    
    test_loss = 0 # 초기 loss를 0으로 설정
    # correct = 0 -> 분류가 아니라 예측이기 때문에 정확도 계산할 필요 없음
    
    for step, (x_batch, y_batch) in enumerate(loader):
        
        hidden = model.initHidden()

        for cell, target in zip(x_batch, y_batch):
            
            cell = cell.unsqueeze(0)

            output, hidden = model(cell, hidden) 
        
        loss = criterion(output.squeeze(0), target)

        test_loss += loss
        
    test_loss /= len(list(loader))

    return test_loss


# 실행

for epoch in range(10):

    train(model, criterion, train_loader)
    
    train_loss = evaluate(model, criterion, train_loader)
    # 학습 데이터의 loss 계산

    val_loss = evaluate(model, criterion, val_loader)
    # 검증 데이터의 loss 계산
    
    print(f'epoch: {epoch}, train loss: {train_loss}, val_loss : {val_loss}')


# test

# test 데이터의 loss 계산
test_loss = evaluate(model, criterion, test_loader)
print(f"test loss : {test_loss}")


# 너무 오래 걸려서 epoch를 10으로 한 결과 2번째 epoch에서 loss가 크게 튀어서 다시 계산해주어야 하는데
# epoch가 충분하지 못해서 학습을 하다가 멈춰버렸고, 결과적으로 test_loss가 높게 나왔습니다.
# 이를 해결하기 위해서 도시빅데이터 서버를 이용해서 epoch 수를 늘린 후 다시 학습을 진행해볼 계획입니다.