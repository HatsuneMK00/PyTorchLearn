# -*- coding: utf-8 -*-
# created by makise, 2022/6/21
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils import data
from maraboupy import Marabou, MarabouCore

from gtsrb.fnn_model_1 import SmallDNNModel
from gtsrb.gtsrb_dataset import GTSRB
from gtsrb.occlusion_layer_extend.occlusion_layer_v2 import OcclusionLayer


def get_a_test_image():
    image = Image.open("../../data/GTSRB/trainingset/00000/00000_00000.ppm")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ])
    tensor = transform(image)
    return tensor


def save_extended_model():
    model = SmallDNNModel()
    model.load_state_dict(torch.load('../../model/fnn_model_gtsrb_small.pth', map_location=torch.device('cpu')))
    image = get_a_test_image()
    # add a new custom layer OcclusionLayer in front of the model
    occlusion_layer = OcclusionLayer(image=image, occlusion_size=(2, 2))
    extended_model = nn.Sequential(
        occlusion_layer,
        *list(model.children())
    )

    # save extended model to pth format
    torch.save(extended_model.state_dict(), '../../model/extended/fnn_model_gtsrb_small_extended_v2.pth')

def save_extended_model_onnx():
    model = SmallDNNModel()
    model.load_state_dict(torch.load('../../model/fnn_model_gtsrb_small.pth', map_location=torch.device('cpu')))
    image = get_a_test_image()
    # add a new custom layer OcclusionLayer in front of the model
    occlusion_layer = OcclusionLayer(image=image, occlusion_size=(2, 2))
    extended_model = nn.Sequential(
        occlusion_layer,
        *list(model.children())
    )
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = torch.Tensor([1, 1])
    onnx_model_filename = 'fnn_model_gtsrb_small_extended_v2_tmp.onnx'
    torch.onnx.export(extended_model, dummy_input, '../../model/extended/' + onnx_model_filename)



def restore_extended_model():
    model = SmallDNNModel()
    fake_image = torch.ones(3, 32, 32)
    occlusion_layer = OcclusionLayer(image=fake_image, occlusion_size=(2, 2))
    extended_model = nn.Sequential(
        occlusion_layer,
        *list(model.children())
    )
    extended_model.load_state_dict(torch.load('../../model/extended/fnn_model_gtsrb_small_extended_v2.pth', map_location=torch.device('cpu')))
    input = torch.Tensor([1, 1])
    output = extended_model(input)
    print(output)


def restore_extended_model_onnx():
    ort_session = onnxruntime.InferenceSession('../../model/extended/fnn_model_gtsrb_small_extended_v2_tmp.onnx')
    input_name = ort_session.get_inputs()[0].name
    input = np.array([1.5, 1.5]).astype(np.float32)
    output = ort_session.run(None, {input_name: input})
    print(output)


def restore_and_run_with_marabou():
    network = Marabou.read_onnx('../../model/extended/fnn_model_gtsrb_small_extended_v2.onnx')
    input = np.array([1, 1]).astype(np.float32)
    output = network.evaluate(input, useMarabou=True)
    print(output)

def restore_and_run_with_marabou_baseline():
    network = Marabou.read_onnx('../../model/fnn_model_gtsrb_small.onnx')
    input = get_a_test_image()
    input = input.numpy()
    input = input.reshape(1, 3, 32, 32)
    output = network.evaluate(input, useMarabou=True)
    print(output)


def restore_and_verify_with_marabou():
    label = 2

    network = Marabou.read_onnx('../../model/extended/fnn_model_gtsrb_small_extended_v2_20.onnx')
    inputs = network.inputVars[0]
    print(inputs.shape)
    outputs = network.outputVars
    print(outputs.shape)
    n_outputs = outputs.flatten().shape[0]
    print(n_outputs)
    network.setLowerBound(inputs[0], 8)
    network.setUpperBound(inputs[0], 9)
    network.setLowerBound(inputs[1], 8)
    network.setUpperBound(inputs[1], 9)

    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs[i], outputs[label]], [1, -1], 0)
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

    vals = network.solve(verbose=True)
    print("verification end")
    print("vals 0" + vals[0])
    print("vals 1")
    print(vals[1])


if __name__ == '__main__':
    # save_extended_model()
    # restore_extended_model()
    # save_extended_model_onnx()
    # restore_extended_model_onnx()
    restore_and_run_with_marabou()
    # restore_and_verify_with_marabou()
    # restore_and_run_with_marabou_baseline()
    # load pytorch model from model/fnn_model_gtsrb_small.pth
    # model = SmallDNNModel()
    # model.load_state_dict(torch.load('../../model/fnn_model_gtsrb_small.pth', map_location=torch.device('cpu')))
    # # generate random input of size (3, 32, 32), type is torch.Tensor
    # input = torch.Tensor([1.5, 1.5])
    # fake_image = torch.ones(3, 32, 32)
    # image = get_a_test_image()
    # # add a new custom layer OcclusionLayer in front of the model
    # occlusion_layer = OcclusionLayer(image=image, occlusion_size=(2, 2))
    # extended_model = nn.Sequential(
    #     occlusion_layer,
    #     *list(model.children())
    # )
    # print the summary of the extended model
    # print(extended_model)

    # output = extended_model(input)
    # output_origin = model(image)
    # print(output)
    # print(output_origin)
    #
    # occluded_image = occlusion_layer(input)
    # plt.subplot(1, 2, 1)
    # # convert torch into numpy array
    # occluded_image = occluded_image.numpy()
    # occluded_image = occluded_image.reshape(3, 32, 32).transpose(1, 2, 0)
    # plt.imshow(occluded_image)
    # plt.subplot(1, 2, 2)
    # image = image.numpy()
    # image = image.reshape(3, 32, 32).transpose(1, 2, 0)
    # plt.imshow(image)
    # plt.show()



