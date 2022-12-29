# -*- coding: utf-8 -*-
# created by makise, 2022/7/7

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from matplotlib import pyplot as plt
from maraboupy import Marabou, MarabouCore

from mnist.fnn_model_2 import FNNModel1
from occlusion_layer.occlusion_layer_v3 import OcclusionLayer


def get_a_test_image() -> (torch.Tensor, int):
    test_loader = data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    iter_test_loader = iter(test_loader)
    iter_test_loader.next()
    image, label = iter_test_loader.next()
    image = image.reshape(1, 28, 28)
    # print(image.shape, label.item())
    return image, label.item()


def save_extended_model():
    model = FNNModel1()
    model.load_state_dict(torch.load('../../model/fnn_model_mnist_2.pth', map_location=torch.device('cpu')))
    image, label = get_a_test_image()
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0])
    extended_model = nn.Sequential(
        occlusion_layer,
        *list(model.children())[1:]
    )
    print(extended_model)
    input = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    # occluded_image = occlusion_layer(input)
    # plt.subplot(1, 2, 1)
    # # convert torch into numpy array
    # occluded_image = occluded_image.numpy()
    # occluded_image = occluded_image.reshape(28, 28)
    # plt.imshow(occluded_image)
    # plt.subplot(1, 2, 2)
    # image = image.numpy()
    # image = image.reshape(28, 28)
    # plt.imshow(image)
    # plt.show()
    output = extended_model(input)
    origin_output = model(image)
    print(output)
    print("origin: ")
    print(origin_output)

    torch.save(extended_model.state_dict(), '../../model/extended/v3.2/fnn_model_mnist_2_extended_shrink.pth')


def save_extended_model_onnx():
    model = FNNModel1()
    model.load_state_dict(torch.load('../../model/fnn_model_mnist_2.pth', map_location=torch.device('cpu')))
    image, label = get_a_test_image()
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0])
    extended_model = nn.Sequential(
        occlusion_layer,
        *list(model.children())[1:]
    )
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    onnx_model_filename = 'fnn_model_mnist_2_extended_shrink.onnx'
    torch.onnx.export(extended_model, dummy_input, '../../model/extended/v3.2/' + onnx_model_filename)


def restore_and_run_with_marabou():
    network = Marabou.read_onnx('../../model/extended/v3.2/fnn_model_mnist_2_extended_shrink.onnx')
    input = np.array([1, 1, 1, 1, 0]).astype(np.float32)
    output = network.evaluate(input, useMarabou=True)
    print(output)


def restore_and_verify_with_marabou():
    label = 3

    network = Marabou.read_onnx('../../model/extended/v3.2/fnn_model_mnist_2_extended_shrink.onnx')
    inputs = network.inputVars[0]
    print(inputs.shape)
    outputs = network.outputVars
    print(outputs.shape)
    n_outputs = outputs.flatten().shape[0]
    print(n_outputs)
    network.setLowerBound(inputs[0], 7)
    network.setUpperBound(inputs[0], 14)
    network.setLowerBound(inputs[1], 5)
    network.setUpperBound(inputs[1], 10)
    network.setLowerBound(inputs[2], 7)
    network.setUpperBound(inputs[2], 14)
    network.setLowerBound(inputs[3], 5)
    network.setUpperBound(inputs[3], 10)
    network.setLowerBound(inputs[4], 0)
    network.setUpperBound(inputs[4], 0)

    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs[i], outputs[label]], [1, -1], -1e-6)
    # output_constraints = []
    # for i in range(7):
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
    # print("vals 1")
    # print(vals[1])


if __name__ == '__main__':
    # get_a_test_image()
    # save_extended_model()
    # save_extended_model_onnx()
    # restore_and_run_with_marabou()
    restore_and_verify_with_marabou()