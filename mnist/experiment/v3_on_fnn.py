# -*- coding: utf-8 -*-
# created by makise, 2022/7/14
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from maraboupy import Marabou, MarabouCore

from mnist.fnn_model_1 import FNNModel1
from occlusion_layer.occlusion_layer_v3 import OcclusionLayer
from find_robust_lb import find_robust_lower_bound, determine_robustness, determine_robustness_color_fixed
import signal
import shutil

result = []

def save_extended_model_onnx(image, model):
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0])
    extended_model = nn.Sequential(
        occlusion_layer,
        *list(model.children())[1:]
    )
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    onnx_model_filename = 'tmp/v3/' + 'fnn_model_mnist_2_extended_shrink.onnx'
    torch.onnx.export(extended_model, dummy_input, onnx_model_filename)
    return onnx_model_filename

def cp_model_file(model_filepath):
    model_filepaths = [model_filepath]
    for i in range(1, 4):
        path = 'tmp/v3/' + 'fnn_model_mnist_1_extended_shrink_{}.onnx'.format(i)
        shutil.copy(model_filepath, path)
        model_filepaths.append(path)
    return model_filepaths


def verify_with_marabou(model_filepath, label, a, b, size_a, size_b, color):
    # signal.signal(signal.SIGALRM, handle_timeout)
    print("start verification marabou {} {} {} {} {} {}".format(a, b, size_a, size_b, color, label), flush=True)
    network = Marabou.read_onnx(model_filepath)
    inputs = network.inputVars[0]
    outputs = network.outputVars
    n_outputs = outputs.flatten().shape[0]
    a_lower, a_upper = a
    b_lower, b_upper = b
    size_a_lower, size_a_upper = size_a
    size_b_lower, size_b_upper = size_b
    color_lower, color_upper = color
    network.setLowerBound(inputs[0], a_lower)
    network.setUpperBound(inputs[0], a_upper)
    network.setLowerBound(inputs[1], size_a_lower)
    network.setUpperBound(inputs[1], size_a_upper)
    network.setLowerBound(inputs[2], b_lower)
    network.setUpperBound(inputs[2], b_upper)
    network.setLowerBound(inputs[3], size_b_lower)
    network.setUpperBound(inputs[3], size_b_upper)
    network.setLowerBound(inputs[4], color_lower)
    network.setUpperBound(inputs[4], color_upper)

    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs[i], outputs[label]], [1, -1], -1e-6)

    options = Marabou.createOptions(solveWithMILP=False, verbosity=0)
    # signal.alarm(60)
    vals = network.solve(verbose=True, options=options)
    # signal.alarm(0)

    # print("verification end for label: ", label, flush=True)
    # print("result from marabou is ", vals[0])
    print("vals 1")
    print(vals[1], flush=True)
    return vals[0]

def handle_timeout():
    raise TimeoutError('verify over 1 min')



if __name__ == '__main__':
    test_loader = data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    iter_on_loader = iter(test_loader)
    model = FNNModel1()
    model.load_state_dict(torch.load('../../model/fnn_model_mnist_1.pth', map_location=torch.device('cpu')))
    for i in range(4):
        print("=" * 20)
        print("image {}:".format(i))
        instrument = {}
        is_robust = True

        total_time_start = time.monotonic()
        image, label = iter_on_loader.next()

        if i not in [1, 2, 4, 5, 6, 7, 8, 9]:
            continue

        image = image.reshape(1, 28, 28)
        label = label.item()

        labels = model(image)
        labels = labels[0].argsort(descending=True)
        spurious_labels = []
        for l in labels:
            if l.item() != label:
                spurious_labels.append(l.item())

        print("spurious label: ", spurious_labels)
        save_model_start = time.monotonic()
        model_filepath = save_extended_model_onnx(image, model)
        model_filepaths = cp_model_file(model_filepath)
        save_model_duration = time.monotonic() - save_model_start
        instrument['save_model_duration'] = save_model_duration

        verify_start = time.monotonic()
        robusts, color_times = determine_robustness(6, spurious_labels, model_filepaths, verify_with_marabou)
        verify_duration = time.monotonic() - verify_start
        instrument['verify_duration'] = verify_duration
        instrument['robusts'] = robusts
        instrument['color_times'] = color_times
        print(instrument)
        result.append(instrument)

    with open('result3_tmp.json', 'w') as f:
        json.dump(result, f)
        f.write('\n')
        f.flush()




