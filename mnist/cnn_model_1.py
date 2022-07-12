# -*- coding: utf-8 -*-
# created by makise, 2022/7/7

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data

BATCH_SIZE = 512
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

train_loader = data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)


class CNNModel1(nn.Module):
    def __init__(self):
        super(CNNModel1, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 3)
        self.fc1 = nn.Linear(10 * 10 * 10, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = torch.relu(self.conv1(x)) # 1*28*28 -> 5*24*24
        x = self.pool(x) # 5*24*24 -> 5*12*12
        x = torch.relu(self.conv2(x)) # 5*12*12 -> 10*10*10

        x = x.view(in_size, -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # forward pass
        output = model(data)
        loss = criterion(output, target)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    model = CNNModel1().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)

    try:
        torch.save(model.state_dict(), '../model/cnn_model_mnist_1.pth')
        print("save success")
    except Exception as e:
        print('Exception: ', e)