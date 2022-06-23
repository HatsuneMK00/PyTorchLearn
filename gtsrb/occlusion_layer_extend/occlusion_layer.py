# -*- coding: utf-8 -*-
# created by makise, 2022/2/24

# using pytorch to train a small feedforward neural network on subset of gtsrb dataset.


import torch
import torch.nn as nn

# define the output size of the network
OUTPUT_SIZE = 7

class OcclusionLayer(nn.Module):
    def __init__(self, image):
        super(OcclusionLayer, self).__init__()
        image_channel, image_height, image_width = image.shape
        self.fc1 = OcclusionFirstLayer(size_in=2, size_out=image_height * 2 + image_width * 2)
        self.fc2 = OcclusionSecondLayer(size_in=self.fc1.size_out, size_out=self.fc1.size_out // 2)
        self.fc3 = OcclusionThirdLayer(size_in=self.fc2.size_out, size_out=image_channel * image_width * image_height, image_shape=image.shape)
        self.fc4 = OcclusionFourthLayer(size_in=self.fc3.size_out, size_out=self.fc3.size_out, image=image)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class OcclusionFirstLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        weights, bias = self.init_weights_bias(size_in, size_out)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.matmul(self.weights, x) + self.bias

    def init_weights_bias(self, size_in, size_out):
        weights = torch.zeros(size_out, size_in)
        bias = torch.zeros(size_out)

        # set the weight
        block_size = size_out // 2
        for i in range(size_out // block_size):
            inner_block_size = block_size // 2
            for j in range(block_size // inner_block_size):
                if j == 0:
                    for k in range(inner_block_size):
                        weights[i * block_size + j * inner_block_size + k, 0] = -1
                elif j == 1:
                    for k in range(inner_block_size):
                        weights[i * block_size + j * inner_block_size + k, 0] = 1
                else:
                    raise ValueError("j should be 0 or 1")

        # set the biases
        block_size = size_out // 2
        for i in range(size_out // block_size):
            inner_block_size = block_size // 2
            for j in range(block_size // inner_block_size):
                sign = 1 if j % 2 == 0 else -1
                for k in range(inner_block_size):
                    bias[i * block_size + j * inner_block_size + k] = sign * (k + 1)

        return weights, bias


class OcclusionSecondLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out

        weights, bias = self.init_weights_bias(size_in, size_out)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.matmul(self.weights, x) + self.bias

    def init_weights_bias(self, size_in, size_out):
        weights = torch.zeros(size_out, size_in)
        block_size = size_out // 2
        for i in range(size_out):
            part = i // block_size
            if part == 0:
                weights[i, i] = -1
                weights[i, i + block_size] = -1
            elif part == 1:
                weights[i, i + block_size] = -1
                weights[i, i + block_size * 2] = -1

        # set the biases
        bias = torch.ones(size_out)

        return weights, bias


class OcclusionThirdLayer(nn.Module):
    def __init__(self, size_in, size_out, image_shape):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.image_shape = image_shape
        weights, bias = self.init_weights_bias(size_in, size_out, image_shape)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.matmul(self.weights, x) + self.bias

    def init_weights_bias(self, size_in, size_out, image_shape):
        weights = torch.zeros(size_out, size_in)
        image_channel, image_height, image_width = image_shape
        # set the weights
        for channel in range(image_channel):
            for i in range(size_out // image_channel):
                r, c = i // image_width, i % image_width
                weights[channel * image_height * image_width + i, r] = 1
                weights[channel * image_height * image_width + i, size_in // 2 + c] = 1

        # set the biases
        bias = torch.ones(size_out)
        bias = -bias

        return weights, bias


class OcclusionFourthLayer(nn.Module):
    def __init__(self, size_in, size_out, image):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.image = image
        weights, bias = self.init_weights_bias(size_in, size_out, image)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.matmul(self.weights, x) + self.bias

    def init_weights_bias(self, size_in, size_out, image):
        # assert image is a tensor
        assert isinstance(image, torch.Tensor)
        # flatten image into 1d
        image_flatten = image.view(-1)
        weights = torch.zeros(size_out, size_in)
        for i in range(size_out):
            weights[i, i] = -image_flatten[i]
        bias = torch.ones(size_out)
        # multiply the biases by image_flatten
        bias = bias * image_flatten

        return weights, bias
