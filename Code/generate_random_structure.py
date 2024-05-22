from random import uniform


def generate_numbers_summing_to_100(n):
    numbers = []
    total = 0
    for i in range(n - 1):
        num = uniform(0, 100 - total - (n - i - 1))
        numbers.append(num)
        total += num
    numbers.append(100 - total)

    if sum(numbers) != 100:
        return generate_numbers_summing_to_100(n)

    return numbers
