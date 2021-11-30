# -*- coding: utf-8 -*-
# @Time    : 2021/11/30 12:25
# @Author  : nieyuzhou
# @File    : waston.py
# @Software: PyCharm
from sympy import symbols, Rational


def Watson(n = 5):
    m = 31
    vector_x = symbols('x1:%d' % (n + 1))
    vector_r = []
    vector_t = [Rational(i + 1, 29) for i in range(29)]
    for i in range(m - 2):
        left = 0
        for j in range(2, n + 1):
            left = left + (j - 1) * vector_x[j - 1] * (vector_t[i] ** (j - 2))
        right = 0
        for j in range(1, n + 1):
            right = right + vector_x[j - 1] * (vector_t[i] ** (j - 1))
        vector_r.append((left - right ** 2 - 1) ** 2)
    vector_r.append(vector_x[0] ** 2)  # r30
    vector_r.append((vector_x[1] - vector_x[0] ** 2 - 1) ** 2)  # r31
    func = sum(vector_r)
    return func, vector_x
