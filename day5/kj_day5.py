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


# 0. logit, sigmoid, softmax의 관계
# logit은 0~1의 확률 input으로 받아서 -inf~inf로 변환해주는 함수
# log(y/(1-y))
# logit과 sigmoid는 역함수 관계
aa = torch.tensor([[-20],[0.5],[100]])
aa.shape
torch.logit(aa)
torch.sigmoid(aa)
torch.softmax(aa, dim=0)



# 1. Logit Layer 작성
class Logit(nn.Module):
    def __init__(self):
        super(Logit, self).__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()



# 2. Cross Entropy 함수 작성
def cross_entropy(data, target):
    loss = 0
    for i in range(len(data)):
        p = data[i]
        l = target[i]
        loss += -(l*p.log() + (1-l)*(1-p).log())
    return loss / len(data)



# 3. Logit layer를 이용한 LogisticRegression 모형 작성
class LogisticRegression(Logit):
    def __init__(self):
        super(LogisticRegression, self).__init__()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))



# 4. 학습 코드 작성
def train(model, criterion, train_data, target):
    model.train()
    
    optimizer.zero_grad()
    
    pred = model(train_data)
    loss = criterion(pred, target.view(pred.shape))
    
    loss.backward()
    optimizer.step()


@torch.no_grad()
def eval(model, criterion, train_data, target):
    model.eval()
    
    pred = model(train_data)
    loss = criterion(pred, target.view(pred.shape))
    auc = roc_auc_score(target.detach().numpy(), pred.view(-1).detach().numpy())
        
    return loss, auc

model = LogisticRegression()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(1, 1000+1):
    
    train(model, cross_entropy, x_train, y_train)
    train_loss, train_auc = eval(model, cross_entropy, x_train, y_train)
    test_loss, test_auc = eval(model, cross_entropy, x_test, y_test)
    
    if epoch % 10 == 0:
        print(f'epoch = {epoch}\n\
                train loss = {train_loss.item()}, test loss = {test_loss.item()}\n\
                train auc = {train_auc}, test auc = {test_auc}')