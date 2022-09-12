# -*- coding: utf-8 -*-
# created by makise, 2022/8/26

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
from gtsrb.gtsrb_dataset import GTSRB
from gtsrb.fnn_model_3 import SmallDNNModel3
from occlusion_layer.occlusion_layer_v4 import OcclusionLayer
from task import determine_robustness_with_epsilon

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

def save_extended_model_onnx(image, model):
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0])
    extended_model = ExtendedModel(occlusion_layer, model)
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = (torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.ones(32 + 32) * 0.01)
    onnx_model_filename = 'tmp/v4/' + 'fnn_model_gtsrb_3_extended_shrink.onnx'
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


if __name__ == '__main__':
    # show_occluded_image()
    # baseline_of_marabou()

    # read parameters from command line
    # has --epsilon
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--sort', type=int, default=1)
    args = parser.parse_args()
    epsilon = args.epsilon
    sort = args.sort

    print("sort: ", sort, flush=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ])
    test_data = GTSRB(root_dir='../../data/', train=False, transform=transform, classes=[1, 2, 3, 4, 5, 7, 8])
    test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    iter_on_loader = iter(test_loader)
    model = SmallDNNModel3()
    model.load_state_dict(torch.load('../../model/fnn_model_gtsrb_small_3.pth', map_location=torch.device('cpu')))
    for i in range(30):
        print("=" * 20)
        print("image {}:".format(i))
        instrument = {}
        is_robust = True

        total_time_start = time.monotonic()
        image, label = iter_on_loader.next()

        image = image.reshape(3, 32, 32)
        label = label.item()

        labels = model(image)
        labels = labels[0].argsort(descending=True)
        spurious_labels = []
        if labels[0] != label:
            print("image {} classified wrong".format(i), flush=True)
            continue
        for l in labels:
            if l.item() != label:
                spurious_labels.append(l.item())

        print("spurious label: ", spurious_labels)
        if sort == 0:
            print("user don't want to sort labels", flush=True)
            spurious_labels = []
            for j in range(7):
                if j != label:
                    spurious_labels.append(j)

        save_model_start = time.monotonic()
        model_filepath = save_extended_model_onnx(image, model)
        save_model_duration = time.monotonic() - save_model_start
        instrument['save_model_duration'] = save_model_duration

        verify_start = time.monotonic()
        robust, adversarial_example = determine_robustness_with_epsilon((5, 5), spurious_labels, epsilon, model_filepath, verify_with_marabou)
        verify_duration = time.monotonic() - verify_start
        instrument['verify_duration'] = verify_duration
        instrument['robust'] = robust
        instrument['adversarial_example'] = adversarial_example
        result.append(instrument)
        print(instrument)

    with open(f'result4_{epsilon}_sort_{sort}.json', 'w') as f:
        json.dump(result, f)
        f.write('\n')
        f.flush()