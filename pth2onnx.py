# -*- coding: utf-8 -*-
# created by makise, 2022/2/22

"""
convert pth format network to onnx format network
"""

import torch
from torch.utils import data
from torchvision import transforms
from gtsrb.gtsrb_dataset import GTSRB
from gtsrb.cnn_model_small import SmallCNNModel

import onnx
import onnxruntime

# load pth model from file
device = torch.device("cpu")
model = SmallCNNModel()
model.load_state_dict(torch.load('model/cnn_model_gtsrb_small.pth', map_location=device))
model = model.to(device)

# convert to onnx
torch.onnx.export(model, torch.randn(64, 3, 32, 32), 'model/cnn_model_gtsrb_small.onnx', verbose=True)


# define the same data transform as when the model is trained
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])

# define the test data
test_data = GTSRB(root_dir='data', train=False, transform=data_transform)
# create data loader for evaluating
test_loader = data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)
samples, labels = iter(test_loader).next()
print(samples.shape, labels.shape)

# test onnx model using one batch size test samples
def test_onnx_model(model_path):
    model_onnx = onnx.load(model_path)
    ort_session = onnxruntime.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    torch_out = ort_session.run(None, {input_name: samples.numpy()}) # the torch_out is 1 * 64 * 43
    torch_out = torch.tensor(torch_out[0]) # we want 64 * 43
    _, predicted = torch.max(torch_out, 1)
    acc = (predicted == labels).sum().item()
    print(f'Accuracy of onnx model is: {100.0 * acc / 64} %')

# the accuracy of this model is roughly 93.75%
test_onnx_model('model/cnn_model_gtsrb_small.onnx')
