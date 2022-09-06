# -*- coding: utf-8 -*-
# created by makise, 2022/9/5
import numpy as np
from matplotlib.patches import Rectangle

from occlusion_layer.occlusion_layer_v3 import OcclusionLayer as OcclusionLayerV3
from occlusion_layer.occlusion_layer_v4 import OcclusionLayer as OcclusionLayerV4
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.utils.data as data
from gtsrb.gtsrb_dataset import GTSRB
from interpolation import occlusion
import json

def show_occluded_image_v4(image, input, epsilons):
    image = image.reshape(3, 32, 32)
    occlusion_layer_v4 = OcclusionLayerV4(image, None, False)
    input_tensor = torch.tensor(input)
    epsilons = torch.tensor(epsilons)
    occluded_image = occlusion_layer_v4(input_tensor, epsilons)
    # convert torch into numpy array
    occluded_image = occluded_image.numpy()
    occluded_image = occluded_image.reshape(3, 32, 32)
    occluded_image = occluded_image.transpose(1, 2, 0)
    # clip occluded_image into [0, 1] with numpy
    occluded_image = np.clip(occluded_image, 0, 1)
    plt.imsave('occluded_image.png', occluded_image)
    # set image type as np.float32
    image = image.numpy()
    image = image.reshape(3, 32, 32)
    image = image.transpose(1, 2, 0)
    plt.imsave('original_image.png', image)


def show_occluded_image_v3(image, input, color):
    image = image.numpy()
    image = image.reshape(3, 32, 32)
    image = image.transpose(1, 2, 0)

    occluded_image = occlusion.occlusion_with_interpolation(image, (input[2], input[0]), (input[1], input[3]), color)

    plt.imsave('occluded_image_v3.png', occluded_image)
    plt.imsave('original_image_v3.png', image)
    # image_reshape = image.reshape(3, 32, 32)
    # occlusion_layer = OcclusionLayer(image=image_reshape, first_layer=None, is_cnn=True)
    # input_tensor = torch.tensor([*input, color])
    # occluded_image = occlusion_layer(input_tensor)
    # plt.subplot(1, 2, 1)
    # # convert torch into numpy array
    # occluded_image = occluded_image.numpy()
    # occluded_image = occluded_image.reshape(3, 32, 32)
    # occluded_image = occluded_image.transpose(1, 2, 0)
    # plt.imshow(occluded_image)
    # plt.subplot(1, 2, 2)
    # # set image type as np.float32
    # image = image.numpy()
    # image = image.reshape(3, 32, 32)
    # image = image.transpose(1, 2, 0)
    # plt.imshow(image)
    # plt.show()


def get_sample_image(idx):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ])
    test_data = GTSRB(root_dir='../../data/', train=False, transform=transform, classes=[1, 2, 3, 4, 5, 7, 8])
    test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    iter_on_loader = iter(test_loader)

    # get image at index idx
    for i in range(idx + 1):
        image, label = next(iter_on_loader)
    return image, label


if __name__ == '__main__':
    # image, label = get_sample_image(6)
    # v4_layer_input = []
    # epsilons = []
    # read json object from ./result4.json
    # with open('./result4.json', 'r') as f:
    #     verification_results = json.load(f)
    #     # only consider the image #4
    #     verification_result = verification_results[0]
    #     adversarial_example = verification_result['adversarial_example']
    #     layer_input = [adversarial_example['a'], adversarial_example['size_a'], adversarial_example['b'], adversarial_example['size_b']]
    #     layer_epsilons = adversarial_example['epsilons']
    #     v4_layer_input.append(layer_input)
    #     epsilons.append(layer_epsilons)
    #     print(layer_input)
    #     print(layer_epsilons)
    #
    # show_occluded_image_v4(image, v4_layer_input[0], epsilons[0])


    v3_layer_inputs = [
        [8.0, 3.0, 10.0, 3.0],
        [11.0, 3.0, 9.999999999999996, 3.0],
        [11.0, 3.0, 11.4, 3.0],
        [10.0, 3.0, 13.6, 3.0],
        [13.0, 3.0, 10.99999999999999, 3.0]
    ]
    colors = [0, 0.2, 0.4, 0.6000000000000001, 0.8]
    image, label = get_sample_image(4)
    label = label.item()
    show_occluded_image_v3(image, v3_layer_inputs[3], colors[3])