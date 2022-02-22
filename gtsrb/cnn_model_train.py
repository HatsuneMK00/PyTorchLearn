# -*- coding: utf-8 -*-
# created by makise, 2022/2/18
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
from cnn_model_1 import AlexnetTSR
from cnn_model_1 import OUTPUT_DIM

from gtsrb_dataset import GTSRB
from torchvision import transforms

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define a simple transformation for dataset
data_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
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
output_dim = OUTPUT_DIM

# create data loader for training and validation
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# print the size of training sample
print("train_size:", len(train_dataset))
print("valid_size:", len(valid_dataset))
# print thte size of training loader
print("train loader length: ", len(train_loader))
print("train loader dataset length: ", len(train_loader.dataset))
# print the shape of training sample
samples, labels = iter(train_loader).next()
print(samples.shape, labels.shape)
# print image grid of random training sample
# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# imshow(torchvision.utils.make_grid(samples))


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
        images = images.to(device)
        labels = labels.to(device)

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

    return epoch_loss / len(train_loader.dataset), epoch_acc / len(train_loader.dataset)


# evaluate the model using validation set
def evaluate():
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            epoch_acc += (predicted == labels).sum().item()

        return epoch_loss / len(valid_loader.dataset), epoch_acc / len(valid_loader.dataset)


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

    print(f'Epoch [{epoch}] Train Loss: {train_loss:.4f} Train Acc: {100.0 * train_acc:.4f} Train Time: {train_end_time - train_start_time:.2f}')
    print(f'Epoch [{epoch}] Validation Loss: {valid_loss:.4f} Validation Acc: {100.0 * valid_acc:.4f} Validation Time: {valid_end_time - valid_start_time:.2f}')

print('Finished Training')
try:
    torch.save(model.state_dict(), '../model/cnn_model_gtsrb.pth')
except Exception as e:
    print('Exception: ', e)

# evaluate model using test set
# roughly 97% accuracy
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * output_dim
    class_total = [0] * output_dim

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs, _ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            # in the last batch, the batch size may be smaller than batch_size
            if i < len(labels):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    acc = 100.0 * correct / total
    print(f'Accuracy of the network: {acc} %')

    for i in range(output_dim):
        acc = 100.0 * class_correct[i] / class_total[i]
        print(f'Accuracy of {i} class: {acc} %')
