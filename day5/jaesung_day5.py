import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import wandb
import numpy as np

from itertools import repeat

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#ipython에서의 경고 메세지 무시
import warnings
warnings.filterwarnings('ignore')

# data generation
seed = 0
# torch에서 random seed를 고정
# np.random.seed와 같은 역할
torch.manual_seed(seed)


mean_vec = list(repeat(1, 50)) + list(repeat(3, 50)) #1이 50개, 3이 50개인 리스트
mean = torch.Tensor([mean_vec, list(reversed(mean_vec))]) #mean_vec과 mean_vec 역순으로 돌린 리스트를 묶어서 텐서로
# mean은 결과적으로 2,100 크기의 텐서
cov1 = torch.eye(100) # 대각원소가 1인 100*100 행렬


cov = torch.stack([cov1, cov1], 0) # 행 방향으로 cov1을 두 개 붙임, 2*(100*100) 
distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)
#평균이 mean_vec, reversed(mean_vec)이고 공분산 행렬이 cov1, cov2인 다변량 정규분포(가우시안분포) 생성
#distrib를 통해 random sampling 및 모수 설정 가능

x = distrib.rsample().T #sample이 아니라 rsample을 사용, transpose (100*100*2)
# sample은 확률분포에서 임의 샘플링하기 때문에 무작위이고 비연속적이며 미분 불가 -> 역전파 불가
# rsample은 reparameterization trick을 활용하여 미분이 가능하고 역전파도 가능
beta = torch.rand(2).uniform_(-1, 1) # (-1,1) 유니폼 분포에서 길이가 2인 랜덤 tensor 제공
y = torch.bernoulli(torch.sigmoid(x @ beta)).to(torch.float32) 
# @는 행렬곱 연산
# (100*100)*2 @ 2*1
# sigmoid 함수를 통해 얻은 0과 1 사이의 값을 확률로 치고 0 또는 1의 결과값 얻음
# y = torch.tensor(list(repeat(1., 50)) + list(repeat(0., 50)))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = seed)



# 1. Logit Layer 작성
class Logit(nn.Module):
    def __init__(self):
        super(Logit, self).__init__()
        self.linear = nn.Linear(2,1)

    def forward(self, x):
        x = self.linear(x)
        x = F.sigmoid(x)
        return x


# 2. Cross Entropy 함수 작성

def cross_entropy(input, y):
    return -(input.log()*y + (1-y)*(1-input).log()).mean()


# 3. Logit layer를 이용한 LogisticRegression 모형 작성
class LogisticRegression(Logit):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.logit = Logit()
    
    def forward(self, x):
        x = self.logit(x)
        return x


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
    return loss, auc,pred

criterion = nn.BCELoss()
model = LogisticRegression()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    
    train(model, cross_entropy, x_train, y_train)
    train_loss, train_auc, pred = eval(model, cross_entropy, x_train, y_train)
    test_loss, test_auc, pred = eval(model, cross_entropy, x_test, y_test)
    
    if epoch % 10 == 0:
        print(f'epoch = {epoch}\n\
                train loss = {train_loss.item()}, test loss = {test_loss.item()}\n\
                train auc = {train_auc}, test auc = {test_auc}')


# class는 사용하기 위해서 먼저 선언이 필수!!!!!!!!!!!

# test