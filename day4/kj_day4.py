import wandb
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

from itertools import repeat

# pip install scikit-learn
from sklearn.model_selection import train_test_split # scikit-learn
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore') # 경고 메시지를 무시하는 라이브러리

# 연산에 사용할 device 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(device)

# data generation
seed = 0
torch.manual_seed(seed) # random seed를 고정하기 위한 함수

mean_vec = list(repeat(1, 50)) + list(repeat(3, 50))
len(mean_vec) # 1 50개, 3 50개인 list 생성

mean = torch.Tensor([mean_vec, list(reversed(mean_vec))])
# reversed()는 순서를 뒤집는 함수, 출력하려면 list로 변환해주어야 함
mean.shape # 2,100

cov1 = torch.eye(100) # I 100x100 행렬 생성
cov = torch.stack([cov1, cov1], 0) # cov1을 위아래로 누적
cov.shape # 2,100,100
# stack에서 dim=0 이면 2,100,100 , dim=1 이면 100,2,100 , dim=2 이면 100,100,2 이런식으로 쌓음

distrib = MultivariateNormal(loc=mean, covariance_matrix=cov) # 다변량 정규분포 공부하기!
distrib.batch_shape # 2
distrib.event_shape # 100

x = distrib.rsample().T # rsample은 미분이 가능한 샘플링 함수 (그냥 sample의 경우 안 된다고 함)
x.shape # 100,2
# x에 100,2 만큼 multivariatenormal에서 난수 추출
# T는 transpose를 의미(2,100 -> 100,2)

beta = torch.rand(2).uniform_(-1,1) # U(-1,1)을 따르는 난수 2개 추출
beta.shape # 2

torch.sigmoid(x@beta) # x와 beta를 곱한다음 sigmoid함수에 대입
# @는 행렬곱 연산자
# sigmoid(x) = 1/(1+exp(-x))
y = torch.bernoulli(torch.sigmoid(x @ beta)).to(torch.float32)
# bernoulli는 0~1 의 값을 input으로 받아서 0 or 1로 random으로 output해주는 함수
y.shape # 100

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)
# train_test_split은 train과 test를 분리해주는 함수
# 정확히는 train과 validation을 분리해준다고 보는 것이 맞음
# validation set을 통해 중간중간 내가 학습한 모델을 평가해줌으로써 overfitting을 방지
# test_size는 test set 구성 비율을 의미, 0.2면 전체 dataset의 20%를 test set으로 지정하겠다는 의미
# shuffle은 입력 안 해주면 default가 True임, split을 해주기 이전에 섞을건지 여부를 결정
x_train.shape # 80,2
x_test.shape # 20,2
y_train.shape # 80
y_test.shape # 20


# 1. Logistic Regression 모형 작성
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2,1) # input dim = 2, output dim = 1
         
    def forward(self, z):
        z = self.linear(z)
        z = F.sigmoid(z) # output이 sigmoid를 거치게 함
        # sigmoid의 경우 init에서 정의해도 되지만 F를 사용하기 위해 여기서 사용함
        return z


# 2. train function 코드 작성
def train(model, device, criterion, train_data, target):
    model.train()

    # 이거 for문으로 안 하고 그냥 바로 돌려도 된다고 하심
    # 바로 돌릴 때 loss에서 차원 안 맞는 문제 생기는데 그 때는 .view()사용해서 하면 됨
    for step, data in enumerate(train_data):
        data = data.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(torch.squeeze(pred), target[step])
        # squeeze는 두 변수 간 dim을 통일시키기 위해서 사용
        # model(x_train[step]).shape가 torch.Size([1])
        # y_train[step].shape가 torch.Size([])
        # 따라서 x_train쪽에 squeeze(0)을 사용하거나, y_train쪽에 unsqueeze(0)을 사용해야 계산 가능함
        loss.backward()
        optimizer.step()


# 3. evaluation function 코드 작성
@torch.no_grad()
def eval(model, device, criterion, train_data, target):
    model.eval()

    test_loss = 0
    prediction = []

    for step, data in enumerate(train_data):
        data = data.to(device)

        pred_prob = torch.squeeze(model(data)) # 학습된 모델에 x_train적합 후 y_train과 차원 통일
        loss = criterion(pred_prob, target[step])
        test_loss += loss
        pred = int(pred_prob.item() > 0.5) # 0.5보다 크면 1, 0.5보다 작으면 0
        prediction.append(pred) # 비어있는 리스트에 pred값들 insert
    
    score = roc_auc_score(target.detach().numpy(), np.array(prediction)) 
    # 첫번째 인자로 true값, 두번째 인자로 예측값을 넣어줌
    # 둘 다 array형식으로 넣어주어야 함
    # 위에서 구한 prediction은 list이므로 np.array를 통해 array로 바꾸어줌
    # true값은 y_train으로 tensor형식이므로 .numpy()를 통해 array로 바꾸어줌
    # detach()의 경우 gradient의 전파를 멈추는 역할을 하는 함수인데 이 예제 내에서는 사용을 하는 것과 안 하는 것의 차이가 없음
    test_loss /= len(train_data)
    # 개별 loss의 합을 평균내서 최종적인 loss을 계산

    return score, test_loss # roc_auc_score값과 loss값 반환


# 4. 학습 코드 작성
'''wandb.login(key="8f03fcd26bb2bda17de28881f3248fab2d9010d6")
wandb.init(project="logistic")'''

model = LogisticRegression().to(device)
criterion = nn.BCELoss()
# Binary Cross Entropy Loss의 줄임말로 input으로 sigmoid(f(x)), target 두 개 필요
# 두 input값이 동일한 shape를 가져야하므로 위에서 squeeze 함수를 통해 일치시켜줬음
optimizer = optim.Adam(model.parameters(), lr=0.01) # optimizer로 Adam 사용, learning rate는 0.01로 설정
#optimizer = optim.SGD(model.parameters(), lr=0.01) # optimizer로 SGD(경사하강법) 사용, learning rate는 0.01로 설정


for epoch in range(1000):
    train(model, device, criterion, x_train, y_train)

    train_score, train_loss = eval(model, device, criterion, x_train, y_train)
    test_score, test_loss = eval(model, device, criterion, x_test, y_test)

    '''wandb.log({'epoch': epoch,
                'train score': train_score,
                'test score': test_score,
                'train loss': train_loss,
                'test loss': test_loss})'''

    if epoch % 10 == 0:
        print(f'epoch = {epoch}\n\
                train loss = {train_loss}, test loss = {test_loss}\n\
                train auc = {train_score}, test auc = {test_score}')
