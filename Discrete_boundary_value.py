# -*- coding: utf-8 -*-
# @Time    : 2021/11/30 15:45
# @Author  : nieyuzhou
# @File    : Discrete_boundary_value.py
# @Software: PyCharm
from sympy import symbols, Rational


def DBV(n = 5):
    m = n
    vector_x = symbols('x1:%d' % (n + 1))
    vector_r = []
    h = Rational(1, n + 1)
    vector_t = [Rational(i + 1, n + 1) for i in range(n)]
    for i in range(m):
        if i == 0:  # x_1
            vector_r.append(
                (2 * vector_x[i] - 0 - vector_x[i + 1] + h ** 2 * (vector_x[i] + vector_t[i] + 1) ** 3 / 2) ** 2)
        if i == n - 1:  # x_n
            vector_r.append(
                (2 * vector_x[i] - vector_x[i - 1] - 0 + h ** 2 * (vector_x[i] + vector_t[i] + 1) ** 3 / 2) ** 2)
        else:
            vector_r.append(
                (2 * vector_x[i] - vector_x[i - 1] - vector_x[i + 1] + h ** 2 * (
                        vector_x[i] + vector_t[i] + 1) ** 3 / 2) ** 2)

    DBV = sum(vector_r)
    return DBV, vector_x
