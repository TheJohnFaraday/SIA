import time
import functools


def measure_exec_time(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        initial_timestamp = time.perf_counter_ns()
        r = f(*args, **kwargs)
        execution_time = time.perf_counter_ns() - initial_timestamp

        return r, execution_time

    return wrapper
