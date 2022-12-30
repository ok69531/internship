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


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


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
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)


# 1. Logit layer 작성
class Logit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logit, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight = torch.nn.Parameter(torch.Tensor(input_dim, output_dim), requires_grad = True)
        self.bias = torch.nn.Parameter(torch.Tensor(output_dim), requires_grad = True)
    
    def forward(self, x):
        prob = F.sigmoid(x @ self.weight + self.bias)
        return prob


# 2. cross entropy 함수 작성
def cross_entropy(prob, target):
    l = -torch.mean(target * torch.log(prob) + (1-target) * torch.log(1-prob))
    return l


# 3. Logit layer를 이용한 LogisticRegression 모형 작성
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = Logit(input_dim, output_dim)
        
    def forward(self, x):
        prob = self.layer(x)
        return prob


# 4. 학습 코드 작성
def train(model):
    model.train()
    
    optimizer.zero_grad()
    
    pred = model(x_train)
    loss = cross_entropy(pred, y_train.view(pred.shape))
    
    loss.backward()
    optimizer.step()


@torch.no_grad()
def eval(model, x, y):
    model.eval()
    
    pred = model(x)
    loss = cross_entropy(pred, y.view(pred.shape))
    auc = roc_auc_score(y.cpu().numpy(), pred.view(-1).data.cpu().numpy())
    
    return loss, auc


input_dim = x.size(1)
output_dim = 1
model = LogisticRegression(input_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.005)

for epoch in range(1, 1000+1):
    train(model)
    train_loss, train_auc = eval(model, x_train, y_train)
    test_loss, test_auc = eval(model, x_test, y_test)
    
    if epoch % 10 == 0:
        print(f'epoch = {epoch}\n\
                train loss = {train_loss}, test loss = {test_loss}\n\
                train auc = {train_auc}, test auc = {test_auc}')
