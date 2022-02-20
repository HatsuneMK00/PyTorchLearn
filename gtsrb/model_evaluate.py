# -*- coding: utf-8 -*-
# created by makise, 2022/2/20

import torch
from torchvision import transforms

from cnn_model_1 import AlexnetTSR
from cnn_model_1 import OUTPUT_DIM
from gtsrb_dataset import GTSRB
from torch.utils.data import DataLoader


# load the model from file
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexnetTSR()
model.load_state_dict(torch.load('../model/cnn_model_gtsrb.pth', map_location=device))
model = model.to(device)

# define some hyperparameters
output_dim = OUTPUT_DIM
batch_size = 64

# define customized data_transform
data_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# define the test dataset
test_data = GTSRB(root_dir='../data', train=False, transform=data_transform)

# define the test dataloader
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# evaluate the loaded model
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




