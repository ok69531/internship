from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)
# glob.glob(파일 경로)
# 특정한 패턴이나 확장자를 가진 파일들의 경로나 이름 필요할 때 사용
# 사용자가 제시한 조건에 맞는 파일명을 리스트로 반환
# 정규식은 사용 x, *나 ?같은 와일드카드만 지원

print(findFiles(r'C:\winter\data_day6\data\names\*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'" #모든 알파벳(소,대문자) + 공백 . , ; ' = 57
n_letters = len(all_letters)


#join 함수는 매개변수로 들어온 리스트에 있는 요소 하나하나를 합쳐서 하나의 문자열로 바꾸어 반환하는 함수

#''.join(리스트)
#''.join(리스트)를 이용하면 매개변수로 들어온 ['a', 'b', 'c'] 이런 식의 리스트를 'abc'의 문자열로 합쳐서 반환해주는 함수

#'구분자'.join(리스트)
#'구분자'.join(리스트)를 이용하면 리스트의 값과 값 사이에 '구분자'에 들어온 구분자를 넣어서 하나의 문자열로
#'_'.join(['a', 'b', 'c']) 라 하면 "a_b_c" 와 같은 형태로 문자열을 만들어서 반환


# 유니코드 문자열을 ascii로 변환
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) # 단어에서 모든 음절을 정준분해하여 저장
        if unicodedata.category(c) != 'Mn' #Mn은 Non-spacing(띄어쓰기 없는거)
        and c in all_letters # 결과적으로 띄어쓰기 없고 all_letters에 들어있는 단어들은 음절을 정준분해하여 저장
    )


# 각 언어의 이름 목록인 category_lines 사전 생성
category_lines = {}
all_categories = []

# 파일을 읽고 줄 단위로 분리
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles(r'C:\winter\data_day6\data\names\*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0] 
    #os.path.basename() 파일 이름만 출력
    #os.path.splitext() 확장자만 따로 분류하여 리스트로
    all_categories.append(category)
    lines = readLines(filename) #줄 단위로 읽어가며 반환
    category_lines[category] = lines #언어명 : 이름들 dictionary

n_categories = len(all_categories)

# 이름을 텐서로 변경

import torch

# all_letters로 문자의 주소 찾기, 예시 "a" = 0
def letterToindex(letter):
    return all_letters.find(letter) #find 찾으면 인덱스, 못찾으면 -1

# 검증을 위해서 한개의 문자를 1*n_letters 텐서로 변환
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters) #글자 길이만큼 0인 텐서 생성
    tensor[0][letterToindex(letter)] = 1 #글자에 해당하는 index는 0->1로 변경
    return tensor
    

# 한 줄(이름)을 line_length * 1 * n_letters,
# 또는 One-hot 문자 벡터의 Array로 변경

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToindex(letter)] = 1
    return tensor

#앞에서 만든 letterToindex와 letterToTensor를 이용하여
#원하는 결과 만드는 함수(for 문으로 한 글자 별로 one-hot -> array)
print(letterToTensor('J'))
print(lineToTensor('Jones').size())

# 네트워크 생성
# 교육 목적의 nn.RNN 대신 직접 RNN 사용

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
        self.softmax = nn.LogSoftmax(dim=1) # dim은 계산이 되는 차원
        #log softmax의 사용 이유
        # 결과값들을 확률 개념으로 해석하기 위해 소프트맥스 결과에 log 씌운 것
        # 소프트맥스는 기울기 소실에 취약하기 때문인 것도 있음

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) #cat을 통해 입력, 은닉 합체
        # 1은 열방향으로 합쳐지는 것을 의미
        hidden = self.i2h(combined) # 다시 은닉층으로 들어갈 값의 차원 맞춰줌
        output = self.i2o(combined) # 
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128 #은닉층의 개수 임의로 설정, 성능에 따라 바꿔감, 하이퍼파라미터
rnn = RNN(n_letters, n_hidden, n_categories)
#n_letters는 앞에서 계산한 소문자,대문자,공백, 온점, 쉼표, 세미콜론 합친 문자열 길이 : 57
#n_categories는 문자 종류의 수 (arabic부터 vietnamese까지) 18개


# forward에서 필요한 input과 hidden 정의
input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)

input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden) 
print(output) # 이 때 출력되는 모든 항목은 해당 카테고리의 우도(likelihood)

# 학습
# 학습 준비
# 우도 중 가장 큰 값을 얻어
# 가장 확률이 높은 언어와 번호 반환
def categoryFromOutput(output):
    top_n, top_i = output.topk(1) # 텐서의 가장 큰 값 및 주소
    category_i = top_i[0].item()  # 텐서에서 정수 값으로 변경( item() 활용)
    return all_categories[category_i], category_i #인덱스를 통해 언어와 번호 반환

print(categoryFromOutput(output)) # 오잉 근데 왜 A 넣었는데 'Korean', 11 이지..?

#학습 예시를 얻는 빠른 방법
import random

def randomChoice(l): # 1 아니라 소문자 엘임 l
    return l[random.randint(0, len(l) - 1)] #random.randint는 정해진 범위에서 정수를 난수로 반환
    # 여기선 l의 모든 값중에 한 개를 랜덤하게 반환

def randomTrainingExample():
    category = randomChoice(all_categories) #언어 랜덤하게 선택
    line = randomChoice(category_lines[category]) #언어에 포함된 단어들 랜덤하게 선택
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor #언어와 단어, 텐서로 변환한 값까지 return

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)


# 네트워크 학습
#nn.Logsoftmax가 마지막 계층이므로
#nn.NLLLoss를 손실함수로 사용

criterion = nn.NLLLoss()

# 각 학습 루프의 목적
# 입력과 목표 tensor 생성
# 0으로 초기화된 은닉 상태 생성
# 각 문자를 읽기(다음 문자를 위한 은닉상태 유지)
# 목표와 최종 출력 비교
# 역전파
# 출력과 손실 반환

learning_rate = 0.005 # 이것을 너무 높게 설정하면 발산할 수 있고, 너무 낮으면 학습이 되지 않을 수 있습니다.

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden() #rnn 클래스에서 초기값 은닉층 길이만큼 0으로 된 텐서

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]): #앞에서 한 line씩 받기로 했으니 for문
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 매개변수의 경사도에 학습률을 곱해서 그 매개변수의 값에 더합니다.
    # 이 모델의 옵티마이저 역할
    # 경사하강법
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
        #add_는 안으로 계속해서 더해줌 , alpha는 이 때 곱해지는 값

    return output, loss.item()


# 예시 데이터를 활용한 실행
import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# 도식화를 위한 손실 추적
current_loss = 0
all_losses = []

def timeSince(since): #실행 시간 계산하는 함수
    now = time.time()  #종료시점에 구한 시간에서 시작값으로 넣은 시간을 뺌
    s = now - since
    m = math.floor(s / 60) # 분 구하고
    s -= m * 60 # 분으로 처리된 시간 빼서 초만 남김
    return '%dm %ds' % (m, s)

start = time.time() 

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample() #랜덤으로 뽑은 예시들 사용
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # iter 숫자, 손실, 이름, 추측 화면 출력
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # 현재 평균 손실을 전체 손실 리스트에 추가
    if iter % plot_every == 0: 
        all_losses.append(current_loss / plot_every) # 1000번에 한 번씩 추가
        current_loss = 0

# 결과 도식화
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

# 결과 평가
# 오차 행렬에서 정확한 추측을 추적
# confusion matrix = 오차 행렬
# 훈련을 통한 예측 성능을 측정하기 위해 예측값과 실제값을 비교하기 위한 표
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000 # 훈련할 때 반복 10000번 해서 결과 10000개 있으니까

# 주어진 라인의 출력 반환
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# 예시들 중에 어떤 것이 정확하게 예측되었는지 기록
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    
    confusion[category_i][guess_i] += 1

# 모든 행을 합계로 나누어 정규화
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# 도식 설정
fig = plt.figure() #도화지를 제공
ax = fig.add_subplot(111)  # 전체를 1*1로 나눠서 1개씩 -> 평소랑 똑같음
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 축 설정
ax.set_xticklabels([''] + all_categories, rotation=90) # 가독성을 위해 label 90도 회전
ax.set_yticklabels([''] + all_categories)

# 모든 tick에서 레이블 지정
ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) #그래프 축 눈금 간격 설정(1로)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

# 사용자 입력으로 실행
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line) # 엔터치고 > input line 을 출력
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        # 상위 n_predictions개의 결과를 반환, sort할 방향은 열, 가장 큰 것부터(True)
        # value와 index를 튜플로
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')