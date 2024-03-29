# -*- coding: utf-8 -*-
# created by makise, 2022/7/28
import json

import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from matplotlib import pyplot as plt
from maraboupy import Marabou, MarabouCore

import time
from mnist.fnn_model_3 import FNNModel1
from occlusion_layer.occlusion_layer_v4 import OcclusionLayer
from find_robust_lb import determine_robustness_with_epsilon

result = []

class ExtendedModel(nn.Module):
    def __init__(self, occlusion_layer, origin_model):
        super(ExtendedModel, self).__init__()
        self.occlusion_layer = occlusion_layer
        self.extended_model = nn.Sequential(
        *list(origin_model.children())[1:]
    )

    def forward(self, x, epsilons):
        x = self.occlusion_layer(x, epsilons)
        x = self.extended_model(x)
        return x


def show_occluded_image():
    model = FNNModel1()
    model.load_state_dict(torch.load('../../model/fnn_model_mnist_1.pth', map_location=torch.device('cpu')))
    image, label = get_a_test_image()
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0])
    input = torch.tensor([10.0, 10.0, 10.0, 10.0])
    epsilons =  torch.ones(28 * 28) * -0.1
    occluded_image = occlusion_layer(input, epsilons)
    plt.subplot(1, 2, 1)
    # convert torch into numpy array
    occluded_image = occluded_image.numpy()
    occluded_image = occluded_image.reshape(28, 28)
    plt.imshow(occluded_image)
    plt.subplot(1, 2, 2)
    image = image.numpy()
    image = image.reshape(28, 28)
    plt.imshow(image)
    plt.show()


def save_extended_model_onnx(image, model):
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0])
    extended_model = ExtendedModel(occlusion_layer, model)
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = (torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.ones(28 + 28) * 0.01)
    onnx_model_filename = 'tmp/v4/' + 'fnn_model_mnist_3_extended_shrink.onnx'
    torch.onnx.export(extended_model, dummy_input, onnx_model_filename)
    return onnx_model_filename


def verify_with_marabou(model_filepath, label, a, b, size_a, size_b, epsilon):
    network = Marabou.read_onnx(model_filepath)
    inputs = network.inputVars[0]
    epsilons = network.inputVars[1]
    outputs = network.outputVars
    n_outputs = outputs.flatten().shape[0]
    a_lower, a_upper = a
    b_lower, b_upper = b
    size_a_lower, size_a_upper = size_a
    size_b_lower, size_b_upper = size_b
    network.setLowerBound(inputs[0], a_lower)
    network.setUpperBound(inputs[0], a_upper)
    network.setLowerBound(inputs[1], size_a_lower)
    network.setUpperBound(inputs[1], size_a_upper)
    network.setLowerBound(inputs[2], b_lower)
    network.setUpperBound(inputs[2], b_upper)
    network.setLowerBound(inputs[3], size_b_lower)
    network.setUpperBound(inputs[3], size_b_upper)
    for i in range(len(epsilons)):
        network.setLowerBound(epsilons[i], -epsilon)
        network.setUpperBound(epsilons[i], epsilon)

    # for epsilon in epsilons:
    #     network.setLowerBound(epsilon, -0.5)
    #     network.setUpperBound(epsilon, 0.5)

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

    options = Marabou.createOptions(solveWithMILP=True, verbosity=0)
    vals = network.solve(options=options)
    print("verification end")
    print("vals 0" + vals[0])
    print("vals 1")
    print(vals[1])
    return vals[0], vals[1]


def get_a_test_image():
    test_loader = data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    iter_test_loader = iter(test_loader)
    # iter_test_loader.next()
    # iter_test_loader.next()
    image, label = iter_test_loader.next()
    image = image.reshape(1, 28, 28)
    # print(image.shape, label.item())
    return image.numpy(), label.item()


def baseline_of_marabou():
    def find_epsilon(network, image: np.array, label: int, epsilon: float):
        """
        :param network: MarabouNetwork
        :param image: input image in np array
        :param label: label of the image
        :param epsilon: the maximum epsilon value
        :return: the epsilon founded for this network
        """
        n_inputs = network.inputVars[0].flatten().shape[0]
        inputs_flattened = network.inputVars[0].flatten()
        print("input_shape", network.inputVars[0].flatten().shape)
        n_outputs = network.outputVars[0].flatten().shape[0]
        outputs_flattened = network.outputVars[0].flatten()
        print("output_shape", network.outputVars[0].flatten().shape)
        flattened_image = image.flatten()

        # print(inputs_flattened[0:100])
        # print(outputs_flattened[0:100])

        # get a marabou variable
        # eps = network.getNewVariable()
        # set the upper bound and lower bound for this variable eps
        # network.setLowerBound(eps, 0)
        # network.setUpperBound(eps, epsilon)
        # add the eps to network inputVars
        # network.inputVars = np.array([eps])
        print(network.inputVars[0].shape)

        # iterate through the inputs and set some equalities
        for i in range(n_inputs):
            val = flattened_image[i]
            for i in range(28):
                for j in range(28):
                    if i >= 10 and i < 20 and j >= 10 and j < 20:
                        network.setLowerBound(inputs_flattened[i * 28 + j], val)
                        network.setUpperBound(inputs_flattened[i * 28 + j], val + epsilon)
                    else:
                        network.setLowerBound(inputs_flattened[i * 28 + j], val)
                        network.setUpperBound(inputs_flattened[i * 28 + j], val)

        # iterate through the outputs and set some equalities
        for i in range(n_outputs):
            if i != label:
                network.addInequality([outputs_flattened[i], outputs_flattened[label]], [1, -1], -1e-6)

        vals = network.solve(verbose=1)
        print("vals: ", vals)

    network = Marabou.read_onnx('../../model/fnn_model_mnist_1.onnx')
    image, label = get_a_test_image()
    epsilon = 0.5

    find_epsilon(network, image, 3, epsilon)

if __name__ == '__main__':
    # show_occluded_image()
    # baseline_of_marabou()

    # read parameters from command line
    # has --epsilon
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--sort', type=int, required=True)
    args = parser.parse_args()
    epsilon = args.epsilon
    sort = args.sort

    print("epsilon: ", epsilon)
    print("sort: ", sort, flush=True)

    test_loader = data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    iter_on_loader = iter(test_loader)
    model = FNNModel1()
    model.load_state_dict(torch.load('../../model/fnn_model_mnist_3.pth', map_location=torch.device('cpu')))
    for i in range(30):
        print("=" * 20)
        print("image {}:".format(i))
        instrument = {}
        is_robust = True

        total_time_start = time.monotonic()
        image, label = iter_on_loader.next()

        image = image.reshape(1, 28, 28)
        label = label.item()

        labels = model(image)
        labels = labels[0].argsort(descending=True)
        spurious_labels = []
        for l in labels:
            if l.item() != label:
                spurious_labels.append(l.item())

        if sort == 0:
            print("user don't want to sort labels", flush=True)
            spurious_labels = []
            for j in range(10):
                if j != label:
                    spurious_labels.append(j)

        print("spurious label: ", spurious_labels)
        save_model_start = time.monotonic()
        model_filepath = save_extended_model_onnx(image, model)
        save_model_duration = time.monotonic() - save_model_start
        instrument['save_model_duration'] = save_model_duration

        verify_start = time.monotonic()
        robust, adversarial_example = determine_robustness_with_epsilon((5, 5), spurious_labels, epsilon / 2, model_filepath, verify_with_marabou)
        verify_duration = time.monotonic() - verify_start
        instrument['verify_duration'] = verify_duration
        instrument['robust'] = robust
        instrument['adversarial_example'] = adversarial_example
        result.append(instrument)
        print(instrument)

    with open('result4_fnn3_{}_sort_{}_size5_1028_0124.json'.format(epsilon, sort), 'w') as f:
        json.dump(result, f)
        f.write('\n')
        f.flush()