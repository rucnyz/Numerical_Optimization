# 牛顿方法的python实现
## 使用语言及依赖

- `python`>=3.8
- `sympy`~=1.9
- `numpy`~=1.21.4
- `matplotlib`~=3.5

## 代码文件结构

- `Discrete_boundary_value.py`：为`Discrete_boundary_value`函数提供python实现函数`DBV()`。
- `Watson.py`：为`Watson`函数提供python实现函数`Watson()`。
- `methods.py`：内含本实验用到的所有求导和算法。
    - `get_gradient()`：获得函数一阶偏导。
    - `get_Hessan()`：获得函数二阶导（海塞矩阵）。
    - `golden_search()`：对给定区间进行黄金分割搜索步长。
    - `damped_newton()`：阻尼牛顿法实现。
    - `modified_newton()`：修正牛顿法实现。
    - `quasi_newton()`：拟牛顿法实现，包括`SR1`、`DFP`、`BFGS`方法。
- `utils.py`：包装函数初始化和算法的调用，方便测试。
    - `get_fun()`：得到原函数、导数、海塞矩阵表达式，提供两种原函数选择：`Watson`和`DBV`。
    - `algo_display()`：将定义好的函数在每种算法中运行并储存以及输出结果。
- `run.ipynb`最终的测试结果，接下来将主要介绍此文件中的结果。

