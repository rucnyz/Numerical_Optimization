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


def golden_search(a, b, func, epsilon = 0.001):
    ai, bi = a, b
    i = 0
    while True:
        if bi - ai < epsilon:
            return (ai + bi) / 2
        al = ai + 0.382 * (bi - ai)
        ar = ai + 0.618 * (bi - ai)
        if func(al) < func(ar):
            bi = ar
        else:
            ai = al
        i = i + 1


def damped_newton(f, g, h, x0, eps, maxiter = 5000):  # 阻尼牛顿法  x0初始点  eps精确度
    xk = x0
    gk, Gk = np.array(g(*xk)), h(*xk)
    Gk = np.squeeze(Gk)
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

        alpha = golden_search(0, 1, iter_xk)
        xk = xk + alpha * dk
        gk, Gk = g(*xk), h(*xk)
        delta = np.linalg.norm(gk, ord = 2)
        deltas.append(delta)
        if count % 100 == 0:
            print("第{}次迭代  函数值为{:.4f}  delta为{:.4f}".format(count, f(*xk).item(), delta))
        count += 1
    count -= 1
    print("\n运行结束")
    if delta > eps:
        print("迭代{}次未收敛  最终delta为{:.4f}  总耗时为{:.4f}s".format(count, delta, time.time() - start_time))
    else:
        print("成功收敛！    总计迭代次数{}    总耗时为{:.4f}s".format(count, time.time() - start_time))
        print("收敛点为{}    函数值为{:.4f}    最终delta为{:.4f}".format(str(xk), f(*xk).item(), delta))
    return deltas


def modified_newton(f, g, h, x0, eps, e1 = 0.1, e2 = 0.01, maxiter = 5000):
    xk = x0
    gk, Gk = np.array(g(*xk)), h(*xk)
    Gk = np.squeeze(Gk)
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
        cosx = (dk.T @ gk) / (np.linalg.norm(gk, ord = 2) * np.linalg.norm(dk, ord = 2))
        # step4
        if cosx > e1:
            dk = -dk
        # step5->6
        if np.abs(cosx) < e2:
            dk = -gk

        # dk = -dk  # 非正定反向（和书上有些区别，是判断的Gk半正定性）
        def iter_xk(a):
            xk_1 = xk + a * dk
            return f(*xk_1)

        alpha = golden_search(0, 1, iter_xk)
        xk = xk + alpha * dk
        gk, Gk = np.array(g(*xk)), h(*xk)
        delta = np.linalg.norm(gk, ord = 2)
        deltas.append(delta)
        if count % 100 == 0:
            print("第{}次迭代  函数值为{:.4f}  delta为{:.4f}".format(count, f(*xk).item(), delta))
        count += 1
    count -= 1
    print("\n运行结束")
    if delta > eps:
        print("迭代{}次未收敛  最终delta为{:.4f}  总耗时为{:.4f}s".format(count, delta, time.time() - start_time))
    else:
        print("成功收敛！    总计迭代次数{}    总耗时为{:.4f}s".format(count, time.time() - start_time))
        print("收敛点为{}    函数值为{:.4f}    最终delta为{:.4f}".format(str(xk), f(*xk).item(), delta))
    return deltas


def quasi_newton(f, g, x0, eps, maxiter = 5000, method = 'BFGS'):
    xk = x0
    gk = np.array(g(*xk))
    n = xk.shape[0]
    # Bk = np.eye(n)
    Hk = np.eye(n)
    # 终止条件
    deltas = []
    delta = np.linalg.norm(gk, ord = 2)
    deltas.append(delta)
    count = 0
    start_time = time.time()
    while delta > eps and count <= maxiter:
        dk = -Hk @ gk

        def iter_xk(a):
            xk_1 = xk + a * dk
            return f(*xk_1)

        # 黄金分割法搜索步长
        alpha = golden_search(0, 1, iter_xk)
        gk_, xk_ = gk, xk
        xk = xk + alpha * dk
        gk = np.array(g(*xk))
        sk = xk - xk_
        yk = gk - gk_

        if method == 'SR1':
            # Bk = Bk + (yk - Bk @ sk) @ (yk - Bk @ sk).T / (yk - Bk @ sk).T @ sk
            Hk = Hk + ((sk - Hk @ yk) @ (sk - Hk @ yk).T) / ((sk - Hk @ yk).T @ yk)
        elif method == 'DFP':
            # Bk = Bk + (1 + sk.T @ Bk @ sk / (sk.T @ yk)) * (yk @ yk.T) / (sk.T @ yk) - (
            #         yk @ sk.T @ Bk + Bk @ sk @ yk.T) / (sk.T @ yk)
            Hk = Hk + (sk @ sk.T) / (sk.T @ yk) - (Hk @ yk @ yk.T @ Hk) / (yk.T @ Hk @ yk)
        else:
            # Bk = Bk + (yk @ yk.T) / (yk.T @ sk) - (Bk @ sk @ sk.T @ Bk) / (sk.T @ Bk @ sk)
            Hk = Hk + (1 + (yk.T @ Hk @ yk) / (sk.T @ yk)) * (sk @ sk.T) / (sk.T @ yk) - (
                    sk @ yk.T @ Hk + Hk @ yk @ sk.T) / (sk.T @ yk)
        delta = np.linalg.norm(gk, ord = 2)
        deltas.append(delta)
        if count % 100 == 0:
            print("第{}次迭代  函数值为{:.4f}  delta为{:.4f}".format(count, f(*xk).item(), delta))
        count += 1
    count -= 1
    print("\n运行结束")
    if delta > eps:
        print("迭代{}次未收敛  最终delta为{:.4f}  总耗时为{:.4f}s".format(count, delta, time.time() - start_time))
    else:
        print("成功收敛！    总计迭代次数{}    总耗时为{:.4f}s".format(count, time.time() - start_time))
        print("收敛点为{}    函数值为{:.4f}    最终delta为{:.4f}".format(str(xk), f(*xk).item(), delta))
    return deltas
