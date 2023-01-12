import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

from itertools import repeat

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# multi
# iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
x = torch.Tensor(iris['data'])
y = torch.Tensor(iris['target'])

seed = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = seed)

x_train.shape # 120,4
y_train.shape # 120


# binary
# data generation
seed = 0
torch.manual_seed(seed)

mean_vec = list(repeat(1, 50)) + list(repeat(3, 50))
mean = torch.Tensor([mean_vec, list(reversed(mean_vec))])
cov1 = torch.eye(100)
cov = torch.stack([cov1, cov1], 0)
distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)

x = distrib.rsample().T
beta = torch.rand(2).uniform_(-1, 1)
y = torch.bernoulli(torch.sigmoid(x @ beta)).to(torch.float32)
# y = torch.tensor(list(repeat(1., 50)) + list(repeat(0., 50)))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = seed)


# 1. Linear layer 작성
class Linear(nn.Module):
    def __init__(self, i_size, o_size):
        super(Linear, self).__init__()
        self.i_size = i_size
        self.o_size = o_size

        # Linear모듈을 사용하지 않고, 직접 구현
        self.weight = nn.Parameter(torch.Tensor(i_size, o_size), requires_grad=True)
        # iris data의 경우 weight의 size가 4x3이 됨
        # requires_grad = True로 설정하면, 해당 파라미터를 업데이트 할 수 있게끔 함
        # requires_grad = False로 설정하면, 해당 파라미터 업데이트 못하게 함
        # 위와 비슷한 기능에는 detach와 no_grad 함수가 있음
        # detach는 모델 간에 gradient가 흐르지 않게 해서 특정 모델만 학습시킬 때 사용
        # no_grad도 마찬가지로 with torch.no_grad(): 라 하고 그 안에 모델을 넣으면 그 모델에 대해서는 학습 시키지 말라는 의미
        self.bias = nn.Parameter(torch.Tensor(o_size), requires_grad=True)
        # iris data의 경우 bias의 size가 3이 됨
        # 이 때 차원수가 x*w는 120x3 b는 3이라서 계산이 안되는데 broadcasting으로 인해 자동으로 차원수를 맞춰서 계산해줌 

    def forward(self, x):
        return x @ self.weight + self.bias

# Linear layer가 잘 작동하는지 확인
print(x_train)
print(x_train.size(1))
print(len(y_train.unique()))
linear = Linear(x_train.size(1), len(y_train.unique()))
lin_x = linear(x_train)
print(lin_x.shape)


# 2. 앞서 만든 Linear layer를 이용하여 Logistic regression model 작성
# 단, Logistic Regression은 이중, 다중 분류가 모두 가능하도록 
class LogisticRegression(nn.Module):
    def __init__(self, i_size, o_size):
        super(LogisticRegression, self).__init__()
        self.i_size = i_size
        self.o_size = o_size
        self.layer = Linear(i_size, o_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(self.layer(x))


# 학습 코드 작성
def train(model, criterion, train_data, target):
    model.train()
    
    optimizer.zero_grad()
    
    pred = model(train_data)
    loss = criterion(pred, one_hot_encoding(target, len(target.unique())))
    
    loss.backward()
    optimizer.step()


# target 데이터를 one-hot vector로 만들기 위한 함수
def one_hot_encoding(data, num_classes):

    one_hot = torch.zeros(len(data), num_classes)

    for c in range(len(data)):
        one_hot[c][int(data[c].item())] = 1

    return one_hot


@torch.no_grad()
def eval(model, criterion, train_data, target):
    model.eval()
    
    pred = model(train_data)
    loss = criterion(pred, one_hot_encoding(target, len(target.unique())))
    acc = sum(torch.argmax((pred), 1) == target).item() / len(target)
        
    return loss, acc


x_size = x_train.size(1)
y_size = len(y_train.unique())
model = LogisticRegression(x_size, y_size)
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()


for epoch in range(1, 1000+1):
    
    train(model, criterion, x_train, y_train)
    train_loss, train_auc = eval(model, criterion, x_train, y_train)
    test_loss, test_auc = eval(model, criterion, x_test, y_test)
    
    if epoch % 10 == 0:
        print(f'epoch = {epoch}\n\
                train loss = {train_loss.item()}, test loss = {test_loss.item()}\n\
                train auc = {train_auc}, test auc = {test_auc}')