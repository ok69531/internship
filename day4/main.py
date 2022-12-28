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


# 1. Logistic Regression 모형 작성
class LogisticRegression(nn.Module):
    def __init__(self, p):
        super(LogisticRegression, self).__init__()
        self.p = p
        self.lin = nn.Linear(p, 1)
    
    def forward(self, x):
        prob = nn.functional.sigmoid(self.lin(x))
        return prob
    


# 2. trina function 코드 작성
def train(model, criterion):
    model.train()
    
    optimizer.zero_grad()
    
    pred = model(x_train)
    loss = criterion(pred, y_train.view(pred.shape))
    
    loss.backward()
    optimizer.step()


# 3. evaluation function 코드 작성
@torch.no_grad()
def eval(model, x, y, criterion):
    model.eval()
    
    pred = model(x)
    loss = criterion(pred, y.view(pred.shape))
    auc = roc_auc_score(y.detach().numpy(), pred.view(-1).detach().numpy())
    
    return loss, auc


# 4. 학습 코드 작성
torch.manual_seed(seed)

model = LogisticRegression(x.size(1))
criterion = nn.BCELoss()

model_param_group = []
model_param_group.append({'params': model.parameters()})
optimizer = optim.SGD(model_param_group, lr = 0.005)

for epoch in range(1, 1000+1):
    
    train(model, criterion)
    train_loss, train_auc = eval(model, x_train, y_train, criterion)
    test_loss, test_auc = eval(model, x_test, y_test, criterion)
    
    if epoch % 10 == 0:
        print(f'epoch = {epoch}\n\
                train loss = {train_loss}, test loss = {test_loss}\n\
                train auc = {train_auc}, test auc = {test_auc}')
