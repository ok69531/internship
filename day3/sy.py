# source code from https://github.com/pytorch/examples/blob/main/mnist/main.py

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


# 연산에 사용할 device 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


# load mnist
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('./', train = True, download = True, transform = transform)
test_dataset = datasets.MNIST('./', train = False, download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False)


# 데이터 구조 확인
# 하나의 데이터만 확인해보면 tuple형태로 되어있고 첫 번째 원소는 이미지, 두 번째 원소는 이미지의 true label인 것을 알 수 있음
a = train_dataset[0]
print(a)
plt.imshow(a[0].squeeze(0))


# model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x


def train(model, device, criterion, loader):
    model.train()
    
    for step, (data, target) in enumerate(loader):
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        pred = model(data)
        
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    

@torch.no_grad()
def eval(model, device, criterion, loader):
    model.eval()
    
    test_loss = 0
    correct = 0
    
    for step, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        pred_prob = model(data)
        loss = criterion(pred_prob, target)
        test_loss += loss
        
        pred = pred_prob.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(loader.dataset)
    acc = correct/len(loader.dataset) * 100
    
    return test_loss, acc
    

# fitting
# login wandb and connect the project
wandb.login(key="my-api-key")
wandb.init(project="internship")


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(10):
    train(model, device, criterion, train_loader)
    
    train_loss, train_acc = eval(model, device, criterion, train_loader)
    test_loss, test_acc = eval(model, device, criterion, test_loader)
    
    wandb.log({'epoch': epoch,
               'train loss': train_loss,
               'test loss': test_loss,
               'train acc': train_acc,
               'test acc': test_acc})
    
    print(f'epoch: {epoch}, train loss: {train_loss}, test_loss: {test_loss}, test acc: {test_acc}%')

wandb.finish()
