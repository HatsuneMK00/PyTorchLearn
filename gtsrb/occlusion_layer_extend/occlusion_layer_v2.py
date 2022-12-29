# -*- coding: utf-8 -*-
# created by makise, 2022/2/24

# using pytorch to train a small feedforward neural network on subset of gtsrb dataset.


import torch
import torch.nn as nn

# define the output size of the network
OUTPUT_SIZE = 7

class OcclusionLayer(nn.Module):
    def __init__(self, image, occlusion_size):
        super(OcclusionLayer, self).__init__()
        self.image_channel, self.image_height, self.image_width = image.shape
        self.occlusion_width, self.occlusion_height = occlusion_size
        self.fc1 = OcclusionFirstLayer(size_in=2, size_out=2 * self.occlusion_width)
        self.fc2 = OcclusionSecondLayer(size_in=self.fc1.size_out, size_out=self.fc1.size_out * 2 * self.image_width)
        self.fc3 = OcclusionThirdLayer(size_in=self.fc2.size_out, size_out=self.fc2.size_out // 2, image_shape=image.shape)
        self.fc4 = OcclusionFourthLayer(size_in=self.fc3.size_out, size_out=self.image_height + self.image_width)
        self.fc5 = OcclusionFifthLayer(size_in=self.fc4.size_out, size_out=self.image_channel * self.image_width * self.image_height, image=image)
        self.fc6 = OcclusionSixthLayer(size_in=self.fc5.size_out, size_out=self.fc5.size_out, image=image)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
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

        # set the weight and bias
        block_size = size_out // 2
        for i in range(size_out):
            part = i // block_size
            weights[i, part] = 1
        # set the biases
        for i in range(size_out // block_size):
            for j in range(block_size):
                bias[i * block_size + j] = j

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
        bias = torch.zeros(size_out)

        # set the weights
        block_size = size_out // size_in
        for i in range(size_in):
            inner_block_size = block_size // 2
            for j in range(2):
                if j == 0:
                    for k in range(inner_block_size):
                        weights[i * block_size + j * inner_block_size + k, i] = -1
                elif j == 1:
                    for k in range(inner_block_size):
                        weights[i * block_size + j * inner_block_size + k, i] = 1

        # set the biases
        for i in range(size_in):
            inner_block_size = block_size // 2
            for j in range(2):
                sign = 1 if j % 2 == 0 else -1
                for k in range(inner_block_size):
                    bias[i * block_size + j * inner_block_size + k] = sign * (k + 1)

        return weights, bias


class OcclusionThirdLayer(nn.Module):
    def __init__(self, size_in, size_out, image_shape):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.image_shape = image_shape
        weights, bias = self.init_weights_bias(size_in, size_out)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.matmul(self.weights, x) + self.bias

    def init_weights_bias(self, size_in, size_out):
        weights = torch.zeros(size_out, size_in)
        block_size = self.image_shape[1]
        block_num = size_out // block_size
        for i in range(block_num):
            for j in range(block_size):
                weights[i * block_size + j, i * block_size * 2 + j] = -1
                weights[i * block_size + j, i * block_size * 2 + block_size + j] = -1

        # set the biases
        bias = torch.ones(size_out)

        return weights, bias


class OcclusionFourthLayer(nn.Module):
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

        block_size = size_out // 2
        for i in range(size_out):
            part = i // block_size
            for j in range(size_in // 2):
                weights[part * block_size + j % block_size, part * (size_in // 2) + j] = 1

        return weights, bias

class OcclusionFifthLayer(nn.Module):
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
        weights = torch.zeros(size_out, size_in)

        image_channel, image_height, image_width = image.shape
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

class OcclusionSixthLayer(nn.Module):
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
        weights = torch.zeros(size_out, size_in)
        bias = torch.ones(size_out)

        # image need to be a numpy array
        assert isinstance(image, torch.Tensor)
        # image need to be 3d array
        assert image.ndim == 3

        # flatten image into 1d
        image_flatten = image.flatten()
        for i in range(size_out):
            weights[i, i] = -image_flatten[i]
        # multiply the biases by image_flatten
        bias = bias * image_flatten

        return weights, bias
