# NLP (Natural Language Processing, 자연어 처리)
# RNN (Recurrent Neural Network, 순환 신경망)


# 1. 데이터 준비

from __future__ import unicode_literals, print_function, division
from io import open # 다양한 유형의 I/O(파일 입출력)를 처리하기 위한 장치를 제공함
# I/O의 세가지 유형 : text, binary, raw
import glob # 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환하는 도구
# * : 임의 길이의 모든 문자열을 의미
# ? : 한 자리의 문자를 의미
import os # OS에 의존하는 다양한 기능을 제공하는 모듈
# 파일이나 디렉토리 조작이 가능하고, 파일의 목록이나 경로를 얻거나, 새로운 파일 작성도 가능

def findFiles(path): 
    return glob.glob(path) 

print(findFiles('data/names/*.txt'))
# data/names 디렉토리에는 18개의 텍스트 파일이 있고, 각 파일에는 한 줄에 하나의 이름이 포함되어 있음
# 대부분 로마자로 되어 있으므로 유니코드에서 ASCII로 변환해야함

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
# abcde...WXYZ .,;'
n_letters = len(all_letters) # 57


# 유니코드 문자열을 ASCII로 변환 (참고 : https://stackoverflow.com/a/518232/2809427)
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if (unicodedata.category(c) != "Mn") and (c in all_letters)
    )
# 문자열을 한 글자씩
# category(chr) : 할당된 general category를 문자열로 반환
# normalize(form, unistr) : unistr에 대한 정규화 형식 form을 반환
# Mn : Mark, no spacing (발음 구별기호)

print(unicodeToAscii('Ślusàrski'))

# 각 언어의 이름 목록인 category_lines dictionary 생성
category_lines = {} # 각 언어의 이름 목록
all_categories = [] # 각 언어명

# 파일을 읽고 줄 단위로 분리
def readLines(filename):
    lines = open(filename, encoding = 'utf-8').read().strip().split('\n')
    # strip은 지정한 문자열을 제거하는 메소드, 인자를 주지 않았으므로 공백을 제거
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    # os.path : 파일이나 디렉토리의 존재 확인, 지정한 경로의 파일명 획득, 경로나 파일명의 결합 등의 용도로 사용
    # os.path.splitext() : 지정된 파일의 확장자와 확장자를 제외한 파일명까지의 경로를 리턴함
    # os.path.basename() : 지정된 경로의 파일명을 리턴
    print(category) # 각 언어명이 하나씩 출력해봄
    all_categories.append(category)
    lines = readLines(filename)
    print(lines) # 각 언어별 이름들을 하나씩 출력해봄
    category_lines[category] = lines

print(category_lines)
print(all_categories)

n_categories = len(all_categories) # 18

print(category_lines['Italian'][:5]) # 이탈리아어 이름 5개 불러오기



# 2. 이름을 텐서로 변경

# 하나의 문자를 표현하기 위해, 크기가 1 x n_letters인 One-Hot 벡터를 사용함
# One-Hot 벡터는 현재 문자의 주소에만 1을 값으로 가지고, 나머지는 0으로 채우는 것을 의미함
# ex. c = [0 0 1 0 0 ...]
# 그 다음 단어를 만들기 위해 One-Hot 벡터들을 2차원 행렬 line_length x 1 x n_letters에 결합시킴 
# 이 때, 중간에 있는 1차원은 pytorch에서 모든 것들이 배치에 있다고 가정하기 때문에 발생함, batch size = 1을 의미

import torch

# all_letters로 문자의 주소 찾기, ex. 'a' = 0
def letterToIndex(letter):
    return all_letters.find(letter) 
    # all_letters에서 letter 찾기, 해당 위치의 index반환 (만약 없으면 -1 반환)

# 검증을 위해서 한 개의 문자(한 글자)를 1 x n_letters의 텐서로 변환
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters) # 길이 57의 영벡터
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# 이름(한 줄에 해당)을 line_length x 1 x n_letters
# 또는 One-Hot 문자 벡터의 array로 변경
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J')) # 36번째 원소만 1로 변하고 나머지는 모두 0인 one-hot벡터로 변환
print(letterToTensor('J').shape) # 1x57인 벡터
print(lineToTensor('Jones')) # 5x1x57인 행렬 생성
print(lineToTensor('Jones').shape) # 단어의 길이(글자 수)가 5, batch size가 1, n_letters(문자의 종류의 수)가 57



# 3. 네트워크 생성

# RNN생성은 여러 시간 단계에 걸쳐서 계층의 매개변수를 복제하는 작업을 포함함
# 계층은 hidden state와 gradient를 가짐
# hidden state : 메모리셀이 출력층 방향 또는 다음 시점인 t+1의 자신에게 보내는 값을 의미

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # input size + hidden size만큼의 차원을 hidden size만큼의 차원으로 선형 변환
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # input size + hidden size만큼의 차원을 output size만큼의 차원으로 선형 변환

        self.softmax = nn.LogSoftmax(dim=1)
        # dim=1 : 두번째 차원에 대해서 LogSoftmax를 적용한다는 의미
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        # dim=1 : 두번째 차원을 기준으로 input layer와 hidden layer을 결합

        hidden = self.i2h(combined) # 선형 변환
        output = self.i2o(combined) # 선형 변환

        output = self.softmax(output) # LogSoftmax 적용
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size) # 원래 hidden size만큼 영벡터 생성

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories) # RNN이라는 클래스를 rnn으로 선언해줌
# input_size가 n_letters, hidden_size가 n_hidden, output_size가 n_categories가 됨
# n_letters는 길이 57의 문자 벡터
# n_categories는 언어이름 벡터의 길이 : 18

input = letterToTensor('A')
print(input) # A라는 문자열을 One-Hot 형식의 벡터로 변환해서 입력
hidden = torch.zeros(1, n_hidden)
print(hidden.shape) # 1, 128 인 영벡터
output, next_hidden = rnn(input, hidden)
print(output)
print(next_hidden)

# 매 단계마다 새로운 텐서를 만들지 않게 하기 위해서 letterToTensor 대신 lineToTensor를 사용
input = lineToTensor('Albert')
print(input)
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)
print(output)
print(next_hidden)


# 4. 학습

# 학습 준비

# 학습으로 들어가기 전에 몇몇의 도움되는 함수를 만들어야 함
# categoryFromOutput은 우리가 알아낸 각 카테고리의 우도인 네트워크 출력을 해석하기 위한 함수
def categoryFromOutput(output):
    top_n, top_i = output.topk(1) # 텐서의 가장 큰 값 및 주소를 반환
    print(top_i)
    #print(top_n)
    category_i = top_i[0].item() # 텐서에서 정수 값으로 변경
    print(category_i)    
    return all_categories[category_i], category_i
    # 텐서의 가장 큰 값의 주소를 통해 all_categories에서 가장 큰 category와 주소를 반환

print(categoryFromOutput(output))


# 학습 예시(하나의 이름과 그 언어)를 얻는 빠른 방법
import random

random.randint(0,len(all_categories)-1) # 0부터 17사이의 임의의 정수
random.randint(0,len(category_lines[category])-1) # 0부터 231사이의 임의의 정수

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(): # 랜덤하게 학습시키기
    category = randomChoice(all_categories) 
    # 0부터 17사이의 임의의 정수에 해당하는 index를 가진 언어명
    line = randomChoice(category_lines[category])
    # 0부터 231사이의 임의의 정수에 해당하는 index를 가진 언어에 속한 이름
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    # 위에서 뽑은 category를 텐서로 만들어줌
    line_tensor = lineToTensor(line)
    # 위에서 뽑은 line을 텐서로 만들어줌
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
# 언어명과 이름을 서로 무관하게 섞어서 학습시킴


# 네트워크 학습

criterion = nn.NLLLoss() # Negative Log Likelihood Loss, L(y)
# 어떤 입력에 대한 softmax 출력 중 가장 큰 값을 y라 할 때 L(y) = -log(y)
# softmax의 출력이 낮은 값은 Loss가 크도록 만들 수 있고, 높은 값은 0에 가깝도록 만들 수 있음
# RNN의 마지막 계층이 Log Softmax이므로 loss function으로 NLLLoss가 적합함
# CrossEntropyLoss는 LogSoftmax와 NLLLoss가 하나의 single class에 합쳐져 있는 것임

# 학습 함수(train function) 생성
# 학습 과정(각 학습 루프)은 다음과 같음
# 입력과 목표 텐서 생성 -> 0으로 초기화된 hidden state 생성 -> 다음 문자를 위한 hidden state 유지하면서 각 문자 읽기
# 목표와 최종 출력을 비교 -> backward 수행 -> 출력과 손실을 반환

learning_rate = 0.005 # 학습률을 너무 높게 설정하면 발산할 수도 있음, 반대로 너무 낮게 설정하면 학습이 아예 되지 않을 수도 있음

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden() # 0으로 초기화된 hidden state 생성

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]): # 현재 line_tensor.size()가 5,1,57이므로 [0]을 적용하면 5가 나옴
        output, hidden = rnn(line_tensor[i], hidden) # 이름의 한 글자씩 rnn에 input으로 넣어줌

    loss = criterion(output, category_tensor) # loss 계산
    loss.backward()

    # 매개변수의 gradient에 learning rate를 곱해서 그 매개변수의 값에 더합니다.
    # 경사하강법을 직접 구현
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


# 예시 데이터를 사용해서 실행
# train function으로 output과 loss를 얻어서 도식화를 위한 loss를 추적

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

# 도식화를 위한 손실 추적
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1): # epoch 100000개
    category, line, category_tensor, line_tensor = randomTrainingExample() # random으로 training할 데이터 가져옴
    output, loss = train(category_tensor, line_tensor) # 가져온 데이터 학습
    current_loss += loss # loss의 합 계산

    # iter 숫자, 손실, 이름, 추측 화면 출력
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        # 예측에 성공하면 check, 실패하면 X
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # 현재 평균 손실을 전체 손실 리스트에 추가
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every) # 현재의 평균 손실을 추가
        current_loss = 0 # 평균 손실 한 번 넣으면 현재 손실을 0으로 초기화


# 결과 도식화

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses) # loss function 도식화


# 결과 평가

# confusion행렬에서 정확한 추측을 추적
confusion = torch.zeros(n_categories, n_categories)
# 18x18 0행렬 생성
n_confusion = 10000

# 주어진 라인의 출력 반환
# 아래는 evaluate function
def evaluate(line_tensor):
    hidden = rnn.initHidden() # hidden layer 0으로 초기화

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# 예시들 중에 어떤 것이 정확하게 예측되었는지 기록
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    # 가장 정확하다고 예측한 category와 그 index를 반환
    category_i = all_categories.index(category)
    # category의 실제 index를 category_i에 저장
    confusion[category_i][guess_i] += 1
    # 최종 index를 confusion matrix에 기록

# 모든 행을 합계로 나누어 정규화
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# 도식 설정
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 축 설정
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# 모든 tick에서 레이블 지정
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
# 주축에서 벗어난 밝은 점을 선택하여 잘못 추측한 언어를 표시할 수 있음



# 사용자 입력으로 실행

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        # 텐서가 가장 큰 값 3개와 그 index를 반환
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')