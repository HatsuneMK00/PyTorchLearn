# -*- coding: utf-8 -*-
# created by makise, 2022/8/4

import json
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from maraboupy import Marabou, MarabouCore

from gtsrb.fnn_model_3 import SmallDNNModel3
from gtsrb.gtsrb_dataset import GTSRB
from occlusion_layer.occlusion_layer_v3 import OcclusionLayer
from task import determine_robustness_color_fixed, determine_robustness

result = []

def save_extended_model_onnx(image, model):
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0])
    extended_model = nn.Sequential(
        occlusion_layer,
        *list(model.children())[1:]
    )
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    onnx_model_filename = 'tmp/v3/' + 'fnn_model_gtsrb_3_extended_shrink.onnx'
    torch.onnx.export(extended_model, dummy_input, onnx_model_filename)
    return onnx_model_filename

def verify_with_marabou(model_filepath, label, a, b, size_a, size_b, color):
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
    network.setLowerBound(inputs[5], color_lower)
    network.setUpperBound(inputs[5], color_upper)
    network.setLowerBound(inputs[6], color_lower)
    network.setUpperBound(inputs[6], color_upper)

    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs[i], outputs[label]], [1, -1], -1e-6)

    options = Marabou.createOptions(solveWithMILP=False, verbosity=0)
    # signal.alarm(60)
    vals = network.solve(options=options)
    # signal.alarm(0)

    print("verification end for label: ", label, flush=True)
    # print("result is ", vals[0])
    return vals[0]


if __name__ == '__main__':
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

        total_time_start = time.monotonic()
        image, label = iter_on_loader.next()

        image = image.reshape(3, 32, 32)
        label = label.item()

        labels = model(image)
        labels = labels[0].argsort(descending=True)
        if labels[0] != label:
            print("image {} classified wrong".format(i), flush=True)
            continue
        spurious_labels = []
        for l in labels:
            if l.item() != label:
                spurious_labels.append(l.item())

        print("spurious label: ", spurious_labels)
        save_model_start = time.monotonic()
        model_filepath = save_extended_model_onnx(image, model)
        save_model_duration = time.monotonic() - save_model_start
        instrument['save_model_duration'] = save_model_duration

        verify_start = time.monotonic()
        robusts, size_times = determine_robustness_color_fixed((1, 10), spurious_labels, model_filepath, verify_with_marabou)
        verify_duration = time.monotonic() - verify_start
        instrument['verify_duration'] = verify_duration
        instrument['robusts'] = robusts
        instrument['size_times'] = size_times
        print(instrument)
        result.append(instrument)

    with open('result3.json', 'w') as f:
        json.dump(result, f)
        f.write('\n')
        f.flush()