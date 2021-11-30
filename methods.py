import time

import numpy as np
from sympy import *


# 得到一阶导
def get_gradient(fexpr, x_vec, show = False):
    grad_expr = [diff(fexpr, x) for x in x_vec]
    if show:
        print('function:')
        pprint(fexpr, use_unicode = True)
        print('gradient:')
        pprint(grad_expr, use_unicode = True)
    return grad_expr


# 得到二阶导
def get_Hessan(fexpr, xvec, show = False):
    Hessan_expr = hessian(fexpr, xvec)
    if show:
        print('Hessian:')
        pprint(Hessan_expr, use_unicode = True)
    return Hessan_expr


def interval_check(fun, step = 1):
    # 适用于一元f(x_k+alpha*d_k)，目的是找到一个先减小后增加的区间
    if fun(0) < fun(step):
        return 0, step
    key = 0
    while fun(key + step) < fun(key):
        key += step
    # 在[key-step,key,key+step]这段是先减小后增加，支持黄金分割
    return key - step, key + step


def golden_section_search(fun, eps = 1e-5, interval = None):
    if interval is None:  # 没给U型区间就需要找
        interval = interval_check(fun)
    interval = list(interval)  # 元组换成列表
    key = (np.sqrt(5) - 1) / 2  # 0.618
    len_interval = interval[1] - interval[0]
    right = interval[0] + key * len_interval
    left = interval[1] - key * len_interval
    while len_interval > eps:  # 黄金分割数值精确搜索
        len_interval *= key
        if fun(left) <= fun(right):
            interval[1], right, left = right, left, right - key * len_interval
        else:
            interval[0], left, right = left, right, left + key * len_interval
    return (interval[1] + interval[0]) / 2


def damped_newton(f, g, G, x0, eps, maxiter = 5000):  # 阻尼牛顿法  x0初始点  eps精确度
    xk = x0
    gk, Gk = np.array(g(*xk)), G(*xk)
    deltas = []
    delta = np.linalg.norm(gk, ord = 2)
    deltas.append(delta)
    count = 0
    start_time = time.time()
    while delta > eps and count <= maxiter:
        try:
            dk = -np.linalg.inv(Gk) @ gk
        except:
            print("Gk奇异，迭代结束")
            return None

        def iter_xk(a):
            xk_1 = xk + a * dk
            return f(*xk_1)

        alpha = golden_section_search(iter_xk, interval = (0, 1))
        xk = xk + alpha * dk
        gk, Gk = g(*xk), G(*xk)
        delta = np.linalg.norm(gk, ord = 2)
        deltas.append(delta)
        if count % 500 == 0:
            print("第{}次迭代  函数值为{:.2f}  delta为{:.2f}".format(count, f(*xk).item(), delta))
        count += 1
    count -= 1
    print("\n运行结束")
    if (delta > eps):
        print("迭代{}次未收敛  最终delta为{:.2f}  总耗时为{:.2f}s".format(count, delta, time.time() - start_time))
    else:
        print("成功收敛！    总计迭代次数{}    总耗时为{:.2f}s".format(count, time.time() - start_time))
        print("收敛点为{}  函数值为{:.2f}".format(str(xk), f(*xk).item()))
    return xk, deltas


def is_positive_def(A):
    try:
        np.linalg.cholesky(A)  # 利用cholesky分解是否可行来判断半正定
        return True
    except:
        return False


def modified_newton(f, g, G, x0, eps, e1 = 0.1, e2 = 0.01, maxiter = 5000):
    xk = x0
    gk, Gk = np.array(g(*xk)), G(*xk)
    # 终止条件
    deltas = []
    delta = np.linalg.norm(gk, ord = 2)
    deltas.append(delta)
    count = 0
    start_time = time.time()
    while delta > eps and count <= maxiter:
        try:
            # 牛顿法
            dk = -np.linalg.inv(Gk) @ gk
        except:
            # 奇异情况采用负梯度
            dk = -gk
        cos = (dk.T @ gk) / (np.linalg.norm(gk, ord = 2)
                             * np.linalg.norm(dk, ord = 2))

        if cos > e1:
            dk = -dk  # step4
        if np.abs(cos) < e2:
            dk = -gk  # step5->6

        # dk = -dk  # 非正定反向（和书上有些区别，是判断的Gk半正定性）
        def iter_xk(a):
            xk_1 = xk + a * dk
            return f(*xk_1)

        alpha = golden_section_search(iter_xk, interval = (0, 1))
        xk = xk + alpha * dk
        gk, Gk = np.array(g(*xk)), G(*xk)
        delta = np.linalg.norm(gk, ord = 2)
        deltas.append(delta)
        if count % 500 == 0:
            print("第{}次迭代  函数值为{:.2f}  delta为{:.2f}".format(count, f(*xk).item(), delta))
        count += 1
    count -= 1
    print("\n运行结束")
    if (delta > eps):
        print("迭代{}次未收敛  最终delta为{:.2f}  总耗时为{:.2f}s".format(count, delta, time.time() - start_time))
    else:
        print("成功收敛！    总计迭代次数{}    总耗时为{:.2f}s".format(count, time.time() - start_time))
        print("收敛点为{}  函数值为{:.2f}".format(str(xk), f(*xk).item()))
    return xk, deltas


def quasi_newton(f, g, x0, eps, maxiter = 5000, method = 'BFGS'):
    xk = x0
    gk = np.array(g(*xk))
    n = xk.shape[0]
    Hk = np.diag(np.ones(n))
    # 终止条件
    deltas = []
    delta = np.linalg.norm(gk, ord = 2)
    deltas.append(delta)
    count = 0
    start_time = time.time()
    while delta > eps and count <= maxiter:
        dk = -Hk @ gk

        def iter_xk(a):  # 线搜索
            xk_1 = xk + a * dk
            return f(*xk_1)

        ak = golden_section_search(iter_xk, interval = (0, 1))
        gk_, xk_ = gk, xk
        xk = xk + ak * dk
        gk = np.array(g(*xk))
        sk = xk - xk_
        yk = gk - gk_

        if method == 'SR1':
            Hk = Hk + ((sk - Hk @ yk) @ (sk - Hk @ yk).T) / ((sk - Hk @ yk).T @ yk)
        elif method == 'DFP':
            Hk = Hk + (sk @ sk.T) / (sk.T @ yk) - (Hk @ yk @ yk.T @ Hk) / (yk.T @ Hk @ yk)
        else:
            Hk = Hk + (1 + (yk.T @ Hk @ yk) / (sk.T @ yk)) * (sk @ sk.T) / (sk.T @ yk) - (
                    sk @ yk.T @ Hk + Hk @ yk @ sk.T) / (sk.T @ yk)
        delta = np.linalg.norm(gk, ord = 2)
        deltas.append(delta)
        if (count % 500 == 0):
            print("第{}次迭代  函数值为{:.2f}  delta为{:.2f}".format(count, f(*xk).item(), delta))
        count += 1
    count -= 1
    print("\n运行结束")
    if (delta > eps):
        print("迭代{}次未收敛  最终delta为{:.2f}  总耗时为{:.2f}s".format(count, delta, time.time() - start_time))
    else:
        print("成功收敛！    总计迭代次数{}    总耗时为{:.2f}s".format(count, time.time() - start_time))
        print("收敛点为{}  函数值为{:.2f}".format(str(xk), f(*xk).item()))
    return xk, deltas
