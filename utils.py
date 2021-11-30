# -*- coding: utf-8 -*-
# @Time    : 2021/11/30 12:50
# @Author  : nieyuzhou
# @File    : utils.py
# @Software: PyCharm
from sympy import init_printing, lambdify
from methods import get_gradient, get_Hessan, modified_newton, damped_newton, quasi_newton
from waston import Watson
from Discrete_boundary_value import DBV
import numpy as np
from IPython.core.interactiveshell import InteractiveShell


# 得到原函数、导数、海塞矩阵表达式
def get_fun(fun_name = "Watson", _n = 5):
    assert fun_name in ["Watson", "DBV"]
    f_sym, vector_x = eval(fun_name)(_n)
    f = lambdify(vector_x, f_sym, 'numpy')
    g_sym = get_gradient(f_sym, vector_x)
    g = lambdify(vector_x, g_sym, 'numpy')
    G_sym = get_Hessan(f_sym, vector_x)
    G = lambdify(vector_x, G_sym, 'numpy')
    return f, g, G


def algo_display(f, g, h, x, eps = 1e-2):
    print("阻尼牛顿法：")
    x_d = damped_newton(f, g, h, x, eps = eps, maxiter = 5000)
    print("---------------------------------------------------------------")
    print("修正牛顿法：")
    x_m = modified_newton(f, g, h, x, eps = eps, maxiter = 5000)
    print("---------------------------------------------------------------")
    print("拟牛顿法：SR1")
    x_S = quasi_newton(f, g, x, eps = eps, maxiter = 5000, method = 'SR1')
    print("---------------------------------------------------------------")
    print("拟牛顿法：DFP")
    x_D = quasi_newton(f, g, x, eps = eps, maxiter = 5000, method = 'DFP')
    print("---------------------------------------------------------------")
    print("拟牛顿法：BFGS")
    x_B = quasi_newton(f, g, x, eps = eps, maxiter = 5000, method = 'BFGS')
    print("---------------------------------------------------------------")
    return x_d, x_m, x_S, x_D, x_B


if __name__ == '__main__':
    n = 5
    # 正式开始
    x0 = np.zeros((n, 1))
    fun, grad, Hessan = get_fun("Watson", 5)
    print("Watson函数使用各算法对比：\n")
    algo_display(fun, grad, Hessan, x0)

    # ---------------

    x0 = np.array([((i + 1) / (n + 1)) * ((i + 1) / (n + 1) - 1) for i in range(n)])
    x0 = x0.reshape((n, 1))
    fun, grad, Hessan = get_fun("DBV", 5)
    print("DBV函数使用各算法对比：\n")
    algo_display(fun, grad, Hessan, x0)
