def percentage(a, b):
    return a / b * 100


def only_positive_values(t):
    return all(value >= 0 for value in t)
