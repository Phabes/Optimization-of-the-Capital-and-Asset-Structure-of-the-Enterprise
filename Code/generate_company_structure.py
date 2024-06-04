from random import uniform
import numpy as np


def generate_random_structure(n):
    numbers = []
    total = 0
    for i in range(n - 1):
        num = uniform(0, 100 - total - (n - i - 1))
        numbers.append(num)
        total += num
    numbers.append(100 - total)

    if sum(numbers) != 100:
        return generate_random_structure(n)

    return numbers


def generate_structure_mean(average_values, deviation, seed=None):
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(None)

    num_elements = len(average_values)

    while True:
        values = np.random.normal(loc=average_values, scale=deviation, size=num_elements)
        if np.all(values > 0):
            break
    values = values / np.sum(values) * 100

    return values
