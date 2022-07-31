# -*- coding: utf-8 -*-
# created by makise, 2022/7/26
import concurrent.futures
import time

import pebble


def determine_robustness(size, labels, model, task):
    colors = [0, 0.2, 0.4, 0.6, 0.8, 1]
    robusts = []
    for c in range(5):
        position_range = (1 ,28 - size + 1)
        step = (position_range[1] - position_range[0] + 1) // 4
        position = [1, step, 2 * step, 3 * step, position_range[1]]
        robust = True
        for label in labels:
            start_time = time.monotonic()
            if not robust:
                break
            with pebble.ProcessPool(1) as pool:
                color = (colors[c], colors[c + 1])
                for i in range(4):
                    a = (position[i], position[i + 1])
                    b = (position[i], position[i + 1])
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
    return robusts

def find_robust_lower_bound(l, u, labels, model, task):
    lower = l
    _, image_height, image_width = (1, 28, 28)
    upper = u
    upper_last_sat = upper
    iter_count = 0
    while upper - lower >= 1:
        iter_count += 1
        position_range = (1, image_height - lower + 1)
        step = (position_range[1] - position_range[0] + 1) // 4
        position = [1, step, 2 * step, 3 * step, position_range[1]]
        robust = True
        for label in labels:
            start_time = time.monotonic()
            if not robust:
                break
            with pebble.ProcessPool(1) as pool:
                for i in range(4):
                    a = (position[i], position[i + 1])
                    b = (position[i], position[i + 1])
                    size_a = (lower, upper)
                    size_b = (lower, upper)
                    future = pool.schedule(task, (model, label, a, b, size_a, size_b, (0, 0)))
                    try:
                        print('iteration {}: label {}, a&b {}, size_a&b {}'.format(iter_count, label, a, size_a), flush=True)
                        task_start_time = time.monotonic()
                        result = future.result(timeout=60)
                        print("task end in ", time.monotonic() - task_start_time)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        print("iteration {} timeout".format(iter_count), flush=True)
                        result = 'unsat' # To keep format with return value of task
                    if result == 'sat':
                        robust = False
                        break
            print("label {} end in {}".format(label, time.monotonic() - start_time))
        if not robust:
            print("size {} is not robust".format((lower, upper)), flush=True)
            upper_last_sat = upper
            upper = (upper + lower) // 2
        else:
            print("size {} is robust".format((lower, upper)), flush=True)
            lower = upper
            upper = (upper_last_sat + lower) // 2
    return upper


def determine_robustness_color_fixed(sizes, labels, model, task):
    robusts = []
    for size in range(sizes[0], sizes[1] + 1):
        size_start_time = time.monotonic()
        position_range = (1, 28 - size + 1)
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
    return robusts
