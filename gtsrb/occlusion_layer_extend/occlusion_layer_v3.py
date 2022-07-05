# -*- coding: utf-8 -*-
# created by makise, 2022/2/24

# using pytorch to train a small feedforward neural network on subset of gtsrb dataset.


import torch
import torch.nn as nn

# define the output size of the network
OUTPUT_SIZE = 7


class OcclusionLayer(nn.Module):
    def __init__(self, image, occlusion_color, first_layer):
        super(OcclusionLayer, self).__init__()
        image_channel, image_height, image_width = image.shape
        self.fc1 = OcclusionFirstLayer(size_in=4, size_out=image_height * 2 + image_width * 2)
        self.fc2 = OcclusionSecondLayer(size_in=self.fc1.size_out, size_out=self.fc1.size_out // 2)
        self.fc3 = OcclusionThirdLayer(size_in=self.fc2.size_out, size_out=image_width * image_height, image_shape=image.shape)
        self.fc4 = OcclusionFourthLayer(size_in=self.fc3.size_out, size_out=image_channel * image_width * image_height, image=image, occlusion_color=occlusion_color, model_first_layer=first_layer)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
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
        block_size = size_out // 4
        for i in range(4):
            if i == 0 or i == 2:
                for j in range(block_size):
                    weights[i * block_size + j, i] = 1
                    bias[i * block_size + j] = -(j + 1)
            elif i == 1 or i == 3:
                for j in range(block_size):
                    weights[i * block_size + j, i - 1] = -1
                    weights[i * block_size + j, i] = -1
                    bias[i * block_size + j] = j + 2

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
        for i in range(2):
            for j in range(block_size):
                weights[i * block_size + j, 2 * i * block_size + j] = -1
                weights[i * block_size + j, 2 * i * block_size + block_size + j] = -1
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
        bias = torch.zeros(size_out)
        _, image_height, image_width = image_shape
        input_block_size = size_in // 2
        block_size = size_out
        # output has only 1 part for occlusion
        for i in range(block_size):
            r, c = i // image_width, i % image_width
            weights[i, r] = 1
            weights[i, input_block_size + c] = 1

        bias = -torch.ones(block_size)

        return weights, bias


class OcclusionFourthLayer(nn.Module):
    def __init__(self, size_in, size_out, image, occlusion_color, model_first_layer):
        super().__init__()
        self.size_in = size_in
        self.size_out = model_first_layer.out_features
        self.image = image
        weights, bias = self.init_weights_bias(size_in, size_out, image, occlusion_color)
        weights = torch.matmul(model_first_layer.weight, weights)
        bias = model_first_layer.bias + torch.matmul(model_first_layer.weight, bias)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.matmul(self.weights, x) + self.bias

    def init_weights_bias(self, size_in, size_out, image, occlusion_color):
        # assert image is a tensor
        assert isinstance(image, torch.Tensor)
        # flatten image into 1d
        image_flatten = image.view(-1)
        image_channel, image_height, image_width = image.shape
        weights = torch.zeros(size_out, size_in)
        for channel in range(image_channel):
            for i in range(size_out // image_channel):
                weights[channel * image_height * image_width + i, i] = occlusion_color - image_flatten[
                    channel * image_height * image_width + i]
        bias = torch.ones(size_out) * image_flatten

        return weights, bias
