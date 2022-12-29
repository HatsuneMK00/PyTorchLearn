# -*- coding: utf-8 -*-
# created by makise, 2022/9/1
import random
import time

import pebble
from concurrent.futures import TimeoutError, CancelledError

def dummy_verify(seq):
    sleep_time = [6, 6, 3, 3, 3, 3, 3, 3]
    time.sleep(sleep_time[seq])
    print(seq)
    return 'sat'

def dummy_timeout_verify():
    time.sleep(6)
    return 'sat'

def dummy_unsat_verify():
    time.sleep(3)
    return 'unsat'

if __name__ == '__main__':
    with pebble.ProcessPool(4) as pool:
        params = [0, 1, 2, 3, 4, 5, 6, 7]
        future = pool.map(dummy_verify, params, chunksize=1, timeout=5)
        iterator = future.result()
        while True:
            try:
                print("iterate next result")
                result = next(iterator)
                print(result)
                if result == 'sat':
                    future.cancel()
            except StopIteration:
                print("stop")
                break
            except TimeoutError as error:
                print("timeout")
                print(error.args)
            except CancelledError as error:
                print("canceled")
            except Exception:
                print("error")

