# -*- coding: utf-8 -*-
# created by makise, 2022/2/27

# This script is used to find the epsilon value with the Marabou solver.
# It's an attempt to find the epsilon value for the given network and how to use Marabou framework.

import onnx
import onnxruntime
from maraboupy import Marabou, MarabouNetwork
import numpy as np
from PIL import Image

from marabou_utils import load_network, load_sample_image


def find_epsilon(network:MarabouNetwork, image:np.array, label:int, epsilon:float):
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

    print(inputs_flattened[0:100])
    print(outputs_flattened[0:100])

    # get a marabou variable
    #eps = network.getNewVariable()
    # set the upper bound and lower bound for this variable eps
    #network.setLowerBound(eps, 0)
    #network.setUpperBound(eps, epsilon)
    # add the eps to network inputVars
    #network.inputVars = np.array([eps])
    print(network.inputVars[0].shape)

    # iterate through the inputs and set some equalities
    for i in range(n_inputs):
        val = flattened_image[i]
        network.setLowerBound(inputs_flattened[i], val - epsilon)
        network.setUpperBound(inputs_flattened[i], val + epsilon)

    # iterate through the outputs and set some equalities
    for i in range(n_outputs):
        if i != label:
            network.setUpperBound(outputs_flattened[i], outputs_flattened[label])

    vals = network.solve(verbose=1)
    print("vals: ", vals)


if __name__ == '__main__':
    # load the sample image
    np_img = load_sample_image()

    # load the network
    network = load_network('../model/fnn_model_gtsrb_small.onnx')
    label = 0
    epsilon = 0.5

    # conduct the find_epsilon function
    find_epsilon(network, np_img, label, epsilon)
