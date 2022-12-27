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
# class LogisticRegression(nn.Module):
    


# 2. trina function 코드 작성
# def train():
    
    

# 3. evaluation function 코드 작성
# @torch.no_grad()
# def eval():


# 4. 학습 코드 작성
# model = 
# criterion = nn.BCELoss()
# optimizer = 
# for epoch in range(1000):