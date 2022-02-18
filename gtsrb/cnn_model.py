# -*- coding: utf-8 -*-
# created by makise, 2022/2/18
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision

from gtsrb_dataset import GTSRB
from torchvision import transforms

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define a simple transformation for dataset
data_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# define the source of training and test data
train_data = GTSRB(root_dir='../data', train=True, transform=data_transform)
test_data = GTSRB(root_dir='../data', train=False, transform=data_transform)

# divide the dataset into training and validation set
ratio = 0.8
train_size = int(ratio * len(train_data))
valid_size = len(train_data) - train_size
train_dataset, valid_dataset = data.random_split(train_data, [train_size, valid_size])

# define hyper parameters
batch_size = 64
epochs = 10
output_dim = 43

# create data loader for training and validation
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# print the size of training sample
print("train_size:", len(train_dataset))
print("valid_size:", len(valid_dataset))
# print the shape of training sample
samples, labels = iter(train_loader).next()
print(samples.shape, labels.shape)
# print image grid of random training sample
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(samples))


# define the CNN model
class AlexnetTSR(nn.Module):
    def __init__(self):
        super(AlexnetTSR, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 1000),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.size(0), -1)
        x = self.classifier(h)
        return x, h


# initialize the CNN model
model = AlexnetTSR().to(device)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# train the model
def train():
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images.to(device)
        labels.to(device)

        # forward pass
        outputs, hidden = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        epoch_acc += (predicted == labels).sum().item()

        # if (i + 1) % 2000 == 0:
        #     print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


# evaluate the model using validation set
def evaluate():
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images.to(device)
            labels.to(device)

            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            epoch_acc += (predicted == labels).sum().item()

        return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader)


# perform training
train_losses = [0] * epochs
train_accs = [0] * epochs
valid_losses = [0] * epochs
valid_accs = [0] * epochs

for epoch in range(epochs):
    print(f'-------------------Epoch [{epoch}]---------------------')
    train_start_time = time.monotonic()
    train_loss, train_acc = train()
    train_end_time = time.monotonic()

    valid_start_time = time.monotonic()
    valid_loss, valid_acc = evaluate()
    valid_end_time = time.monotonic()

    train_losses[epoch] = train_loss
    train_accs[epoch] = train_acc
    valid_losses[epoch] = valid_loss
    valid_accs[epoch] = valid_acc

    print(f'Epoch [{epoch}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Train Time: {train_end_time - train_start_time:.2f}')
    print(f'Epoch [{epoch}] Validation Loss: {train_loss:.4f} Validation Acc: {train_acc:.4f} Validation Time: {train_end_time - train_start_time:.2f}')

print('Finished Training')
torch.save(model.state_dict(), './model/cnn_model.pth')

# evaluate model using test set
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * output_dim
    class_total = [0] * output_dim

    for images, labels in test_loader:
        images.to(device)
        labels.to(device)

        outputs, _ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                class_correct += 1
            class_total += 1

    acc = 100.0 * correct / total
    print(f'Accuracy of the network: {acc} %')

    for i in range(output_dim):
        acc = 100.0 * class_correct[i] / class_total[i]
        print(f'Accuracy of {i} class: {acc} %')
