#!/usr/bin/env python

import math


def entropy(ps):
    e = 0
    for p in ps:
        e += -p * math.log(p)
    print(e)

    e = 1
    for p in ps:
        e -= -p * math.log(p) / math.log(len(ps))
    print(e)


def get_data(values, numbers):
    print(values, numbers)
    n = len(values)
    a = []
    for i in range(n):
        a += [1.0 * values[i] / numbers[i] for _ in range(numbers[i])]
    # print(a)
    return a


print('--------------------------------------------------------')
a = get_data([1.0], [2000])
print(len(a))
print(sum(a))
entropy(a)

print('--------------------------------------------------------')
a = get_data([0.1, 0.9], [1000, 1000])
print(len(a))
print(sum(a))
entropy(a)

print('--------------------------------------------------------')
a = get_data([0.1, 0.9], [1900, 100])
# print(a)
print(len(a))
print(sum(a))
entropy(a)

print('--------------------------------------------------------')
a = get_data([0.1, 0.9], [1990, 10])
# print(a)
print(len(a))
print(sum(a))
entropy(a)

print('--------------------------------------------------------')
a = get_data([0.1, 0.9], [1999, 1])
# print(a)
print(len(a))
print(sum(a))
entropy(a)

print('--------------------------------------------------------')
a = get_data([0.2, 0.8], [1000, 1000])
print(len(a))
print(sum(a))
entropy(a)

print('--------------------------------------------------------')
a = get_data([0.2, 0.8], [1900, 100])
print(len(a))
print(sum(a))
entropy(a)

print('--------------------------------------------------------')
a = get_data([0.2, 0.8], [1990, 10])
print(len(a))
print(sum(a))
entropy(a)

print('--------------------------------------------------------')
a = get_data([0.2, 0.8], [1999, 1])
print(len(a))
print(sum(a))
entropy(a)
