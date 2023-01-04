from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import pandas as pd

def findFiles(path): return glob.glob(path)

print(findFiles(r'C:\winter\data\archive\*.csv'))


# 각 언어의 이름 목록인 category_lines 사전 생성
data_values = {}
variable_names = []


for filename in findFiles(r'C:\winter\data\archive\*.csv'):
    variable_name = os.path.splitext(os.path.basename(filename))[0] 
    variable_names.append(variable_name)
    data_values[variable_name] = pd.read_csv(filename) 

df = data_values['PJME_hourly']

# 시각화
import plotly.graph_objs as go
from plotly.offline import iplot

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




df = df.set_index(['Datetime'])
df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()
    
df = df.rename(columns={'PJME_MW': 'value'})
plot_dataset(df, title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')

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


# datetime을 통해 hour부터 week_of_year 까지의 5개의 feature 추가로 할당
df_features = (
                df
                .assign(hour = df.index.hour)
                .assign(day = df.index.day)
                .assign(month = df.index.month)
                .assign(day_of_week = df.index.dayofweek)
                .assign(week_of_year = df.index.week)
              )

# one-hot encoding
def onehot_encode_pd(df, col_name):
    dummy_df_list = []
    for col in col_name:
        dummies = pd.get_dummies(df[col], prefix=col)
        dummy_df_list.append(dummies)
    temp_df = pd.concat(dummy_df_list, axis=1)
    return pd.concat([df, temp_df], axis=1)#.drop(columns=col_name)


df_features = onehot_encode_pd(df_features, ['month','day','day_of_week','week_of_year'])
df_features

# 주기적 시간 특성 생성
import numpy as np
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

#휴일 추가

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

#data 분리
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

# 스케일러

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()
    
scaler = get_scaler('robust')

# dataloader

import torch
from torch.utils.data import TensorDataset, DataLoader

batch_size = 64

train_features = torch.Tensor(X_train_arr)
train_targets = torch.Tensor(y_train_arr)
val_features = torch.Tensor(X_val_arr)
val_targets = torch.Tensor(y_val_arr)
test_features = torch.Tensor(X_test_arr)
test_targets = torch.Tensor(y_test_arr)

train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

train_targets
train_features[0].shape



# day6 RNN 모형

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        #모델의 구조가 입력층 + 은닉층 -> 결합 -> if i20 -> softmax -> output
        # 입력층 + 은닉층 -> 결합 -> if i2h -> 은닉층
        # self.i2h와 i2o의 입력 크기와 출력 크기는 모델의 형태를 반영
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) #cat을 통해 입력, 은닉 합체
        hidden = self.i2h(combined) # 다시 은닉층으로 들어갈 값의 차원 맞춰줌
        output = self.i2o(combined)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128 #은닉층의 개수 임의로 설정, 성능에 따라 바꿔감, 하이퍼파라미터



def train(model, criterion ,data_loader):    
    for data, target in data_loader: # data, target은 배치
        model.train()

        hidden = model.initHidden()
        
        model.zero_grad()
            
        for i in range(len(data)):
            pred,hidden = model(data[i].view([-1,113]), hidden)
            loss = criterion(pred, target[i].unsqueeze(0))
            
        loss.backward()
    
        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)
    return pred, loss.item()

@torch.no_grad()
def eval(model, criterion, data_loader):
      
    model.eval()
     
    test_loss = 0
    
    for data, target in data_loader:
        hidden = model.initHidden() 
        
            
        for i in range(len(data)):
            pred,hidden = model(data[i].view([-1,113]), hidden)
            loss = criterion(pred, target[i].unsqueeze(0))
            test_loss += loss
        

    test_loss /= len(data_loader.dataset)
   
    
    return test_loss  


model = RNN(113, n_hidden, 1)
learning_rate = 0.01
criterion = nn.MSELoss()
best_valid=  float('inf')
patient=0

for epoch in range(1000):
    train(model, criterion, train_loader)
    
    train_loss=  eval(model, criterion, train_loader)
    val_loss=  eval(model, criterion, val_loader)

    print(f'epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}')
    if val_loss<=best_valid:
        torch.save(model, './assets/model/best_rnn.pth')
        patient=0
        best_valid=val_loss
    else:
        patient+=1
        if patient>=10:
            break






test_loss = eval(model, criterion, test_loader)
print(f"test loss: {test_loss}")