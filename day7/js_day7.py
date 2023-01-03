# 1. Linear layer 작성
# class Linear(torch.nn.Modue:)

# 2. 앞서 만든 Linear layer를 이용하여 Logistic regression model 작성
# Logistic Regression은 이중, 다중 분류가 모두 가능하도록

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

from itertools import repeat

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings('ignore')


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')



# binary data generation
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


# iris data generation
seed = 0
torch.manual_seed(seed)

iris = load_iris()
target_iris = torch.Tensor(iris.target)
data_iris = torch.Tensor(iris.data)
x_train, x_test, y_train, y_test = train_test_split(data_iris, target_iris, test_size = 0.2, random_state = seed)



# 1. Linear layer 작성
class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.w = nn.Parameter(torch.randn(input_size, output_size))
        self.b = nn.Parameter(torch.zeros(output_size))
        


    def forward(self, x):
        return x @ self.w + self.b

# 2. Logistic regression model 작성
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer = Linear(input_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        prob = self.softmax(self.layer(x))
        return prob

input_size = x_train.size(1)
output_size = len(y_train.unique())
model = LogisticRegression(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 3. 모델 학습
def train(model, criterion, train_data, target):
    model.train()
        
    optimizer.zero_grad()
    
    pred = model(train_data)
    
    loss = criterion(pred, target.long()) 
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()

# 4. 모델 평가

@torch.no_grad()
def eval(model, criterion, train_data, target):
    model.eval()
    
    pred = model(train_data)

    loss = criterion(pred, target.long())
    correct_prediction = torch.argmax(pred, 1) == target
    accuracy = correct_prediction.float().mean()
    return loss, accuracy

# 5. 결과
for epoch in range(1000):
    
    train(model, criterion, x_train, y_train)
    train_loss, train_accuracy= eval(model, criterion, x_train, y_train)
    test_loss, test_accuracy= eval(model, criterion, x_test, y_test)
    
    if epoch % 10 == 0:
        print(f'epoch = {epoch}\n\
                train loss = {train_loss.item()}, test loss = {test_loss.item()}\n\
                train accuracy = {train_accuracy}, test accuracy = {test_accuracy}')

