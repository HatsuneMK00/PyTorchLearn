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
from gtsrb.fnn_model_1 import SmallDNNModel

import onnx
import onnxruntime

# define some global parameters for exporting the pytorch model
model_name = 'gtsrb_cnn_small'
use_device = 'cpu'
input_size = (32, 32)
channel_num = 3
output_dim = 43
batch_size = 1
model_path = 'model/cnn_model_gtsrb_small.pth'
onnx_model_path = 'model/cnn_model_gtsrb_small.onnx'  # only used in testing
model_save_dir = 'model/'
only_export = True
only_test = False


def initialize_model(model_name):
    if model_name == 'gtsrb_cnn_small':
        model = SmallCNNModel()
    elif model_name == 'gtsrb_fnn_1':
        model = SmallDNNModel()
    else:
        raise ValueError('model name is not defined')

    return model


def initialize_device():
    device = torch.device('cuda' if torch.cuda.is_available() and use_device == 'gpu' else 'cpu')
    return device


def export_model_2_onnx(model, model_name, device, input_size, channel_num, batch_size, model_path, model_save_dir):
    # load state dict of model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    # export the model to onnx format
    dummy_input = torch.randn(batch_size, channel_num, input_size[0], input_size[1])
    # encode input_size, channel_num, batch_size, output_dim to the onnx model filename
    onnx_model_filename = f'{model_name}_inputSize_{input_size[0]}, {input_size[1]}_channelNum_{channel_num}_batchSize_{batch_size}_outputDim_{output_dim}.onnx'
    torch.onnx.export(model, dummy_input, model_save_dir + onnx_model_filename, verbose=True)


# test onnx model using one batch size test samples
def test_model_onnx(onnx_model_path, input_size, channel_num, output_dim, batch_size):
    # define the same data transform as when the model is trained
    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ])
    # define the test data
    if output_dim != 43:
        test_data = GTSRB(root_dir='data/', train=False, transform=data_transform, classes=range(0, output_dim))
    else:
        test_data = GTSRB(root_dir='data/', train=False, transform=data_transform)
    # create data loader for evaluating
    test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    samples, labels = iter(test_loader).next()
    print("samples shape: ", samples.shape)
    print("labels shape: ", labels.shape)
    # load the onnx model
    onnx_model = onnx.load(onnx_model_path)
    # create the onnxruntime session
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # create the input tensor
    input_name = ort_session.get_inputs()[0].name
    input_tensor = samples.numpy()
    input_tensor = input_tensor.reshape(batch_size, channel_num, input_size[0], input_size[1])
    # run the model
    output_tensor = ort_session.run(None, {input_name: input_tensor})  # the torch_out is 1 * batch_size * output_dim
    output_tensor = torch.tensor(output_tensor[0])
    _, predicted = torch.max(output_tensor, 1)
    acc = (predicted == labels).sum().item() / batch_size
    print(f'accuracy: {100.0 * acc} &')


if __name__ == '__main__':
    if only_test:
        test_model_onnx(onnx_model_path, input_size, channel_num, output_dim, batch_size)
    else:
        model = initialize_model(model_name)
        device = initialize_device()
        export_model_2_onnx(model, model_name, device, input_size, channel_num, batch_size, model_path, model_save_dir)
        if not only_export:
            test_model_onnx(onnx_model_path, input_size, channel_num, output_dim, batch_size)
