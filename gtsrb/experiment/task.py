# -*- coding: utf-8 -*-
# created by makise, 2022/8/4

import concurrent.futures
import time

import pebble

# experiment 2
def determine_robustness(size, labels, model, task):
    colors = [0, 0.2, 0.4, 0.6, 0.8, 1]
    robusts = []
    color_times = []
    for c in range(1):
        color_start_time = time.monotonic()
        position_range = (1 ,32 - size + 1)
        step = (position_range[1] - position_range[0] + 1) // 4
        position = [(step, 2 * step), (2 * step, 3 * step), (1, step), (3 * step, position_range[1])]
        robust = True
        for label in labels:
            start_time = time.monotonic()
            if not robust:
                break
            with pebble.ProcessPool(1) as pool:
                color = (0.1, 0.9)
                for i in range(4):
                    a = position[i]
                    b = position[i]
                    size_a = (size, size)
                    size_b = (size, size)
                    future = pool.schedule(task, (model, label, a, b, size_a, size_b, color))
                    try:
                        print('label {}, a&b {}, size_a&b {}, color {}'.format(label, a, size_a, color), flush=True)
                        task_start_time = time.monotonic()
                        result = future.result(timeout=60)
                        print("task end in ", time.monotonic() - task_start_time)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        print("timeout", flush=True)
                        result = 'unsat' # To keep format with return value of task
                    if result == 'sat':
                        robust = False
                        break
            print("label {} end in {}".format(label, time.monotonic() - start_time))
        robusts.append(robust)
        color_times.append(time.monotonic() - color_start_time)
    return robusts, color_times

def determine_robustness_color_fixed(sizes, labels, model, task):
    robusts = []
    size_times = []
    for size in range(sizes[0], sizes[1] + 1):
        size_start_time = time.monotonic()
        position_range = (1, 32 - size + 1)
        step = (position_range[1] - position_range[0] + 1) // 4
        position = [(step, 2 * step), (2 * step, 3 * step), (1, step), (3 * step, position_range[1])]
        robust = True
        for label in labels:
            start_time = time.monotonic()
            if not robust:
                break
            with pebble.ProcessPool(1) as pool:
                for i in range(4):
                    a = position[i]
                    b = position[i]
                    size_a = (size, size)
                    size_b = (size, size)
                    color = (0, 0)
                    future = pool.schedule(task, (model, label, a, b, size_a, size_b, color))
                    try:
                        print('label {}, a&b {}, size_a&b {}, color {}'.format(label, a, size_a, color), flush=True)
                        task_start_time = time.monotonic()
                        result = future.result(timeout=60)
                        print("task end in ", time.monotonic() - task_start_time)
                        print("result is ", result, flush=True)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        print("timeout", flush=True)
                        result = 'unsat'  # To keep format with return value of task
                    if result == 'sat':
                        robust = False
                        break
            print("label {} end in {}".format(label, time.monotonic() - start_time))
        robusts.append(robust)
        print("size {} end in {}".format(size, time.monotonic() - size_start_time))
        size_times.append(time.monotonic() - size_start_time)
    return robusts, size_times


def determine_robustness_with_epsilon(size, labels, epsilon, model, task):
    robust = True
    adversarial_example = {}
    position_range = (1, 32 - size[0] + 1)
    step = (position_range[1] - position_range[0] + 1) // 4
    position = [(step, 2 * step), (2 * step, 3 * step), (1, step), (3 * step, position_range[1])]
    for label in labels:
        start_time = time.monotonic()
        if not robust:
            break
        with pebble.ProcessPool(1) as pool:
            for i in range(4):
                a = position[i]
                b = position[i]

                size_a = size
                size_b = size
                future = pool.schedule(task, (model, label, a, b, size_a, size_b, epsilon))
                try:
                    print('label {}, a&b {}, size_a&b {}, epsilon {}'.format(label, a, size_a, epsilon), flush=True)
                    task_start_time = time.monotonic()
                    result, result_input = future.result(timeout=60)
                    print("task end in ", time.monotonic() - task_start_time)
                    print("result is ", result, flush=True)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    print("timeout", flush=True)
                    result = 'unsat'  # To keep format with return value of task
                if result == 'sat':
                    robust = False
                    adversarial_example['a'] = result_input[0]
                    adversarial_example['size_a'] = result_input[1]
                    adversarial_example['b'] = result_input[2]
                    adversarial_example['size_b'] = result_input[3]
                    adversarial_example['epsilons'] = [result_input[i] for i in range(4, 4 + 32 + 32)]
                    break
        print("label {} end in {}".format(label, time.monotonic() - start_time), flush=True)
    return robust, adversarial_example