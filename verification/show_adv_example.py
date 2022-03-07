# -*- coding: utf-8 -*-
# created by makise, 2022/3/7

# read result file produced from marabou_occlusion verification and show the adversarial example
# together with the original image
import json

import numpy as np
from matplotlib import pyplot as plt

result_file_dir = '../experiment/results/'
result_filename = ''


def read_result_file(filename):
    with open(filename, 'r') as f:
        # read result file as json
        result = json.load(f)

    return result


def get_adv_examples(result):
    # get the adversarial example
    adv_examples = []
    for i in range(len(result)):
        if not result[i]['robust']:
            predicted_label = result[i]['predicted_label']
            true_label = result[i]['true_label']
            origin_image = result[i]['origin_image']
            adv_example = result[i]['adversarial_example']
            idx = i
            # pack them into dict and append to adv_examples
            adv_example_dict = {'predicted_label': predicted_label,
                                'true_label': true_label,
                                'origin_image': origin_image,
                                'adversarial_example': adv_example,
                                'idx': idx}
            adv_examples.append(adv_example_dict)
    return adv_examples


def show_adv_example(adv_examples, show = range(5)):
    """
    Show the adversarial example together with the original image using plt
    Only show adversarial example in the range defined by show
    :param adv_examples: parsed result from result file
    :param show: range of the adversarial example to show, default is range(5)
    :return:
    """
    # convert adv_examples and origin image to numpy array
    assert len(adv_examples) >= len(show)
    for i in show:
        adv_example = np.array(adv_examples[i]['adversarial_example'])
        origin_image = np.array(adv_examples[i]['origin_image'])
        predicted_label = adv_examples[i]['predicted_label']
        true_label = adv_examples[i]['true_label']
        idx = adv_examples[i]['idx']
        # show the adversarial example
        plt.subplot(1, 2, 1)
        plt.imshow(adv_example)
        plt.title('adversarial example')
        plt.subplot(1, 2, 2)
        plt.imshow(origin_image)
        plt.title('origin image')
        plt.suptitle('predicted label: %d, true label: %d, idx: %d' % (predicted_label, true_label, idx))
        plt.show()


if __name__ == '__main__':
    path = result_file_dir + result_filename
    result = read_result_file(path)
    adv_examples = get_adv_examples(result)
    show_adv_example(adv_examples)
