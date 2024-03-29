# -*- coding: utf-8 -*-
# created by makise, 2022/4/2

"""
This file is used to conduct experiments parallely to find out a better block size.
"""
import json
import os
import time
from concurrent.futures import TimeoutError
import pebble

import numpy as np

import three_marabou_occlusion
import four_marabou_occlusion
from verification.show_adv_example import get_adv_examples

experiment_num = 3
pe_result_dir = "/home/GuoXingWu/occlusion_veri/PyTorchLearn/experiment/results/thought_3/pe_20220409_142644/"
analysis = False

def analyze_result():
    # load all file in the result directory
    ite = os.walk(pe_result_dir)
    results = {}
    for path, _, file_list in ite:
        for file_name in file_list:
            if file_name.endswith(".json"):
                with open(os.path.join(path, file_name), "r") as f:
                    # f is a json file
                    data = json.load(f)
                    adv_examples = get_adv_examples(data)
                    # fixme only use result of the first image
                    results[file_name] = {
                        "total_verify_time": adv_examples[0]['total_verify_time'],
                    }
    # sort the results by total_verify_time
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_verify_time'])

    # print the result
    print("---Result of {}---".format(pe_result_dir))
    for result in sorted_results:
        print("{} : {}".format(result[0], result[1]['total_verify_time']))
    print("------------------------------------------------")




# fixme Due to file I/O (probably), the parallelization is not working. Set processes to 1 on server currently.
if __name__ == '__main__':
    if analysis == True:
        analyze_result()
        exit()
    parameters = []

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

    if experiment_num == 3:
        block_sizes = [(8, 8)]
        occlusion_color = 0
        # occlusion size is a list of tuple (i, i), i ranges from 0 to 31
        occlusion_sizes = [(i, i) for i in range(1, 11)]
        for block_size in block_sizes:
            for occlusion_size in occlusion_sizes:
                parameters.append((occlusion_size, occlusion_color, block_size, timestamp))

        # fixme set processes to 1 on server currently
        # conduct experiment parallely with concurrent.futures
        with pebble.ProcessPool(1) as pool:
            for parameter in parameters:
                future = pool.schedule(three_marabou_occlusion.conduct_experiment, parameter, timeout=60 * 30)
                try:
                    future.result()
                    print("Parallel Experiment: Complete for {}".format(parameter), flush=True)
                except TimeoutError:
                    # future.cancel()
                    print("Parallel Experiment: Timeout for {}".format(parameter), flush=True)
    elif experiment_num == 4:
        block_sizes = [(32, 32)]
        color_epsilon = 0.5
        # occlusion size is a list of tuple (i, i), i ranges from 0 to 31
        occlusion_sizes = [(i, i) for i in range(1, 32)]
        for block_size in block_sizes:
            for occlusion_size in occlusion_sizes:
                parameters.append((occlusion_size, color_epsilon, block_size, timestamp))

        # fixme set processes to 1 on server currently
        # conduct experiment parallely with concurrent.futures
        with pebble.ProcessPool(1) as pool:
            for parameter in parameters:
                future = pool.schedule(four_marabou_occlusion.conduct_experiment, parameter)
                try:
                    future.result(timeout=60 * 60 * 1)
                    print("Parallel Experiment: Complete for {}".format(parameter), flush=True)
                except TimeoutError:
                    future.cancel()
                    print("Parallel Experiment: Timeout for {}".format(parameter), flush=True)

    print("Parallel Experiment: Done!")
