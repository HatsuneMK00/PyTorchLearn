# -*- coding: utf-8 -*-
# created by makise, 2022/2/27

# Some useful functions for Marabou

from maraboupy import Marabou, MarabouNetwork

# load a network from onnx format file and return MarabouNetwork
def load_network(filename) -> MarabouNetwork:
    """
    Load a network from onnx format file and return MarabouNetwork
    :param filename: the name of the onnx file
    :return: network: MarabouNetwork
    """
    # the path of network file
    path = '../model/' + filename
    # load the network
    network = Marabou.read_onnx(path)
    return network