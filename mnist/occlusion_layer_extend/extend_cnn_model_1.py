# -*- coding: utf-8 -*-
# created by makise, 2022/7/7

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from maraboupy import Marabou, MarabouCore

from mnist.cnn_model_1 import CNNModel1
from occlusion_layer.occlusion_layer_v3 import OcclusionLayer

class ExtendedCNNModel1(nn.Module):
    def __init__(self, occlusion_layer, origin_model):
        super(ExtendedCNNModel1, self).__init__()
        self.occlusion_layer = occlusion_layer
        self.origin_model = origin_model

    def forward(self, x):
        x = self.occlusion_layer(x)
        x = x.reshape((1, 1, 28, 28))
        x = self.origin_model(x)
        return x

def get_a_test_image() -> (torch.Tensor, int):
    test_loader = data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    image, label = iter(test_loader).next()
    image = image.reshape(1, 28, 28)
    # print(image.shape, label.item())
    return image, label.item()


def save_extended_model():
    model = CNNModel1()
    model.load_state_dict(torch.load('../../model/cnn_model_mnist_1.pth', map_location=torch.device('cpu')))
    image, label = get_a_test_image()
    occlusion_layer = OcclusionLayer(image=image, occlusion_color=0, first_layer=None, is_cnn=True)
    extended_model = ExtendedCNNModel1(occlusion_layer, model)
    print(extended_model)
    input = torch.tensor([1.0, 1.0, 1.0, 1.0])
    origin_output = model(image.reshape(1, 1, 28, 28))
    output = extended_model(input)
    print(output)
    print("origin: ")
    print(origin_output)

    torch.save(extended_model.state_dict(), '../../model/extended/v3.1/cnn_model_mnist_1_extended.pth')


def save_extended_model_onnx():
    model = CNNModel1()
    model.load_state_dict(torch.load('../../model/cnn_model_mnist_1.pth', map_location=torch.device('cpu')))
    image, label = get_a_test_image()
    occlusion_layer = OcclusionLayer(image=image, occlusion_color=0, first_layer=None, is_cnn=True)
    extended_model = ExtendedCNNModel1(occlusion_layer, model)
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = torch.tensor([1.0, 1.0, 1.0, 1.0])
    onnx_model_filename = 'cnn_model_mnist_1_extended.onnx'
    torch.onnx.export(extended_model, dummy_input, '../../model/extended/v3.1/' + onnx_model_filename)


def restore_and_run_with_marabou():
    network = Marabou.read_onnx('../../model/extended/v3.1/cnn_model_mnist_1_extended.onnx')
    input = np.array([1, 1, 1, 1]).astype(np.float32)
    output = network.evaluate(input, useMarabou=True)
    print(output)


def restore_and_verify_with_marabou():
    label = 3

    network = Marabou.read_onnx('../../model/extended/v3.1/cnn_model_mnist_1_extended.onnx')
    inputs = network.inputVars[0]
    print(inputs.shape)
    outputs = network.outputVars[0]
    print(outputs.shape)
    n_outputs = outputs.flatten().shape[0]
    print(n_outputs)
    network.setLowerBound(inputs[0], 1)
    network.setUpperBound(inputs[0], 5)
    network.setLowerBound(inputs[1], 1)
    network.setUpperBound(inputs[1], 1)
    network.setLowerBound(inputs[2], 1)
    network.setUpperBound(inputs[2], 5)
    network.setLowerBound(inputs[3], 1)
    network.setUpperBound(inputs[3], 1)

    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs[i], outputs[label]], [1, -1], -1e-6)
    # output_constraints = []
    # for i in range(10):
    #     if i == label:
    #         continue
    #     eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    #     eq.addAddend(1, outputs[i])
    #     eq.addAddend(-1, outputs[label])
    #     eq.setScalar(0)
    #     output_constraints.append([eq])
    # network.addDisjunctionConstraint(output_constraints)

    options = Marabou.createOptions(solveWithMILP=False)
    vals = network.solve(verbose=True, options=options)
    print("verification end")
    print("vals 0" + vals[0])
    print("vals 1")
    print(vals[1])


if __name__ == '__main__':
    # get_a_test_image()
    # save_extended_model()
    # save_extended_model_onnx()
    # restore_and_run_with_marabou()
    restore_and_verify_with_marabou()