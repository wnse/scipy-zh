# Introduction

*开篇介绍没有翻译*

为了简洁方便，通常以下面的方式导入主要的包，读者将会看到这些简写贯穿了NumPy和SciPy源码，虽然并不要求但是强烈建议读者也使用这些简写方式。

```python
>>> import numpy as np
>>> import matplotlib as mpl
>>> import matplotlib.pyplot as plt
```



## SciPy Organization

SciPy根据不同类型的科学计算分为子包，如下表。

| 子包        | 描述                   |
| ----------- | ---------------------- |
| cluster     | 聚类算法               |
| contants    | 物理和数学常数         |
| fftpack     | 快速傅利叶变换例程     |
| integrate   | 积分和常微分方程求解器 |
| interpolate | 插值和平滑样条         |
| io          | 输入和输出             |
| linalg      | 线性代数               |
| ndimage     | N-维图像处理           |
| odr         | 正交距离回归           |
| optimize    | 优化和寻根例程         |
| signal      | 信号处理               |
| sparse      | 稀疏矩阵和相关例程     |
| spatial     | 空间数据结构和算法     |
| special     | 特殊函数               |
| stats       | 分布统计和函数         |

SciPy子包需要分别导入，比如

```python
>>> from scipy import linalg, optimize
```

由于一些子包的函数使用频繁，这些在scipy命名空间是可用的，交互式会话和程序使用起来更方便。另外一些numpy中基本的数列函数在scipy的一级包中也是可用的。在查找子包之前可以先看一看这些常用函数。

## Finding Documentation

SciPy和NumPy有各版本的文档，包含HTML和PDF两种格式https://docs.scipy.org/，其中涵盖了几乎所有的函数。然而这个文档仍然编写中，一些部分可能不完整或者不免疏忽。由志愿者组织，依赖于社区成长，所以欢迎任何人的参与，不论是提供反馈或者完善文档或者代码。

Python的文档字符串在SciPy的线上文档中适用。两种方法获取帮助信息。一是Python中的pydoc模块的help命令，不加参数进入帮助交互对话（即```>>>help```）模式允许搜索Python中所有可用的关键字和模块。二是运行以对象为参数的命令*help(obj)*会显示该对象的调用签名和文档字符串。

使用pydoc方法调用```help```比较复杂但是以页面形式展示文本。有时会干扰终端上正在运行交互对话。使用```numpy.info```也可以调用numpy/scipy特定的帮助系统。传给```help```命令的对象签名和文档字符串会打印到标准输出（或者是第三个参数传入的可写对象）。```numpy.info```第二个关键字参数定义了打印输出行的最大宽度。如果一个模块作为参数传给```help```，这个模块定义的函数和类的列表会同时输出。比如：

```
>>> np.info(optimize.fmin)
fmin(func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None,
      full_output=0, disp=1, retall=0, callback=None)

Minimize a function using the downhill simplex algorithm.

Parameters
----------
func : callable func(x,*args)
    The objective function to be minimized.
x0 : ndarray
    Initial guess.
args : tuple
    Extra arguments passed to func, i.e. ``f(x,*args)``.
callback : callable
    Called after each iteration, as callback(xk), where xk is the
    current parameter vector.

Returns
-------
xopt : ndarray
    Parameter that minimizes function.
fopt : float
    Value of function at minimum: ``fopt = func(xopt)``.
iter : int
    Number of iterations performed.
funcalls : int
    Number of function calls made.
warnflag : int
    1 : Maximum number of function evaluations made.
    2 : Maximum number of iterations reached.
allvecs : list
    Solution at each iteration.

Other parameters
----------------
xtol : float
    Relative error in xopt acceptable for convergence.
ftol : number
    Relative error in func(xopt) acceptable for convergence.
maxiter : int
    Maximum number of iterations to perform.
maxfun : number
    Maximum number of function evaluations to make.
full_output : bool
    Set to True if fopt and warnflag outputs are desired.
disp : bool
    Set to True to print convergence messages.
retall : bool
    Set to True to return list of solutions at each iteration.

Notes
-----
Uses a Nelder-Mead simplex algorithm to find the minimum of function of
one or more variables.
```

另外一个很有用的命令是```dir()```，可以用来查找模块或者包中的命名空间，比如：

```
>>> dir(optimize)
['BFGS', 'Bounds', 'HessianUpdateStrategy', 'LbfgsInvHessProduct', 'LinearConstraint', 'NonlinearConstraint', 'OptimizeResult', 'OptimizeWarning', 'RootResults', 'SR1', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '_basinhopping', '_bglu_dense', '_cobyla', '_constraints', '_differentiable_functions', '_differentialevolution', '_dual_annealing', '_group_columns', '_hessian_update_strategy', '_lbfgsb', '_linprog', '_linprog_ip', '_linprog_rs', '_linprog_simplex', '_linprog_util', '_lsap', '_lsap_module', '_lsq', '_minimize', '_minpack', '_nnls', '_numdiff', '_remove_redundancy', '_root', '_root_scalar', '_shgo', '_shgo_lib', '_slsqp', '_spectral', '_trlib', '_trustregion', '_trustregion_constr', '_trustregion_dogleg', '_trustregion_exact', '_trustregion_krylov', '_trustregion_ncg', '_zeros', 'absolute_import', 'anderson', 'approx_fprime', 'basinhopping', 'bisect', 'bracket', 'brent', 'brenth', 'brentq', 'broyden1', 'broyden2', 'brute', 'check_grad', 'cobyla', 'curve_fit', 'diagbroyden', 'differential_evolution', 'division', 'dual_annealing', 'excitingmixing', 'fixed_point', 'fmin', 'fmin_bfgs', 'fmin_cg', 'fmin_cobyla', 'fmin_l_bfgs_b', 'fmin_ncg', 'fmin_powell', 'fmin_slsqp', 'fmin_tnc', 'fminbound', 'fsolve', 'golden', 'lbfgsb', 'least_squares', 'leastsq', 'line_search', 'linear_sum_assignment', 'linearmixing', 'linesearch', 'linprog', 'linprog_verbose_callback', 'lsq_linear', 'minimize', 'minimize_scalar', 'minpack', 'minpack2', 'moduleTNC', 'newton', 'newton_krylov', 'nnls', 'nonlin', 'optimize', 'print_function', 'ridder', 'root', 'root_scalar', 'rosen', 'rosen_der', 'rosen_hess', 'rosen_hess_prod', 'shgo', 'show_options', 'slsqp', 'test', 'tnc', 'toms748', 'zeros']
```



# Basic functions

## Interaction with Numpy

SciPy建立在NumPy之上，对于所有基本的数列处理需求，您可以使用NumPy函数。

```python
>>> import numpy as np
>>> np.some_function()
```

本手册不会给出每个函数的详细描述（这些可以使用NumPy参考指南或者使用```help```，```info```和```source```命令），而会讨论一些更有用的命令，同时需要介绍他们完全的潜能。

为了使用一些SciPy模块的函数，可以这样操作：

```python
>>> from scipy import some_module
>>> some_module.some_function()
```

scipy的一级包中也包含了来自numpy和numpy.lib.scimath的函数。即使如此，最好直接使用numpy调用他们。

### index tricks

有些类实例为极少用到的特殊的函数提供了高效的创建数列的方式。这一部分会讨论numpy.mgrid，numpy.ogrid，numpy.r_和numpy.c\_快速创建数列的操作。

例如，以下的操作：

```python
>>> a = np.concatenate(([3], [0]*5, np.arange(-1, 1.002, 2/9.0)))
```

可以像这样使用r_命令：

```python
>>> a = np.r_[3, [0]*5, -1:1:10j]
```

这样可以减少输入并且易读。请注意对象是怎样联系的，和在构建序列时用的切片语法。另外需要解释的是复数10j在切片中作为step-size的使用。这种非标准化的方式允许数字被解释为序列中需要产生点的数量，而不是step-size（请注意，我们会使用长整数符号10L，但是随着整数变得统一，这种符号在Python中可能会消失）。可能有人对这种非标准化用法不习惯，但是这提供给用户以一种易读的方式快速创建复数向量的操作。当以这种方式指定值的数量时，结束值是包含在内的。

“r”代表行（row）联接，如果以逗号分割的对象是2维数列，它们以行堆叠（因此一定会有列堆叠）。对应命令c_以列堆叠2维数列，与r\_命令操作1维数列一致。

```python
>>> import numpy as np
>>> np.r_[[3]*5, [0]*5]
array([3, 3, 3, 3, 3, 0, 0, 0, 0, 0])
>>> np.c_[[3]*5, [0]*5]
array([[3, 0],
       [3, 0],
       [3, 0],
       [3, 0],
       [3, 0]])
```

另外一个非常有用并且使用扩展切片符号的类实例是mgrid。简单的说，这个函数可以方便的对arange进行替换来创建1维序列。也可以在step-size中使用复数代表两端（包含）数字范围内的数值的个数。但是，此函数的真正目的是生成N，N-D数列，这些数列为N-D体提供坐标数列。最简单的理解方式是看一个例子中的用法。

```python
>>> np.mgrid[0:5,0:5]
array([[[0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4]],

       [[0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]]])
>>> np.mgrid[0:5:4j,0:5:4j]
array([[[0.        , 0.        , 0.        , 0.        ],
        [1.66666667, 1.66666667, 1.66666667, 1.66666667],
        [3.33333333, 3.33333333, 3.33333333, 3.33333333],
        [5.        , 5.        , 5.        , 5.        ]],

       [[0.        , 1.66666667, 3.33333333, 5.        ],
        [0.        , 1.66666667, 3.33333333, 5.        ],
        [0.        , 1.66666667, 3.33333333, 5.        ],
        [0.        , 1.66666667, 3.33333333, 5.        ]]])            
```

拥有这样的网格数列有时非常有用。然而由于NumPy和SciPy的广播规则，并不是经常需要构造N维的网格数列。如果这是生成网格的唯一目的，则应改用ogrid函数，该函数会使用newaxis明智地生成“开放”网格，以创建N，N维数列，其中每个数列中只有一维的长度大于1。如果网格的唯一目的是生成用于评估N维函数的采样点，这将节省内存并创建相同的结果。

```python
>>> np.ogrid[0:5:4j,0:5:4j]
[array([[0.        ],
       [1.66666667],
       [3.33333333],
       [5.        ]]), array([[0.        , 1.66666667, 3.33333333, 5.        ]])]
```

### Shape manipulation

在此类函数中，包括用于从N维数列中挤出一维长度的程序，以确保数组至少为一维，二维或3维，并按行，列和“页面”（在第三维中）堆叠（连接）数列。也提供用于拆分数列的例程（与堆叠数列大致相反）。

### Polynomials

SciPy中有两种方式（可替换）处理1维的多项式。第一种是使用NumPy中的poly1d类。该类接受系数或多项式根来初始化多项式，然后多项式对象可以在代数表达式中操作，积分，微分和计算，甚至像多项式一样输出。

```python
>>> from numpy import poly1d
>>> p = poly1d([3, 4, 5])
>>> print(p)
   2
3 x + 4 x + 5
>>> print(p*p)
   4      3      2
9 x + 24 x + 46 x + 40 x + 25
>>> print(p.integ(k=6))
   3     2
1 x + 2 x + 5 x + 6
>>> print(p.deriv())
6 x + 4
>>> p([4, 5])
array([ 69, 100])
```

处理多项式的另一种方法是作为一个系数数列，其中数列的第一个元素给出了最大幂系数。 有对系数序列的多项式的加，减，乘，除，积分，微分和计算的显式函数。

### Vectorizing functions (vectorize)

NumPy提供的一个特性是类vectorize，它接受标量返回标量，将普通的Python函数转换为一个“矢量化函数”，其广播规则与其他NumPy函数（即通用函数或ufuncs）相同。例如，假设有一个名为```addsubtract```的Python函数定义为：

```python
>>> def addsubtract(a, b):
...     if a > b:
...             return a - b
...     else:
...             return a + b
...
```

这个函数接受两个标量变量，返回一个标量结果，类向量化可以用来“向量化”这个函数。

```python
>>> vec_addsubtract = np.vectorize(addsubtract)
```

返回一个接受数列参数返回数列结果的函数

```python
>>> vec_addsubtract([0,3,6,9],[1,3,5,7])
array([1, 6, 1, 2])
```

这个函数完全可以以向量方式定义，而不用向量化。然而，采用优化或集成例程的函数可能只能使用```vectorize```进行向量化。

### Type handling

请读者注意[numpy.iscomplex](https://docs.scipy.org/doc/numpy/reference/generated/numpy.iscomplex.html#numpy.iscomplex)/[numpy.isreal](https://numpy.org/doc/stable/reference/generated/numpy.isreal.html#numpy.isreal)和[numpy.iscomplexobj](https://docs.scipy.org/doc/numpy/reference/generated/numpy.iscomplexobj.html#numpy.iscomplexobj)/[numpy.isrealobj](https://docs.scipy.org/doc/numpy/reference/generated/numpy.isrealobj.html#numpy.isrealobj)的区别。前者是基于数列的，返回一个由1和0组成的数列，代表每个元素的测试结果，后者是基于对象的，返回标量描述测试整个对象的结果。

经常需要获取复数的实部和/或虚部，虽然复数和数列具有返回这些值的属性，但是如果不清楚对象是不是复数，最好使用numpy.real和numpy.imag函数形式。这些函数可以在任何可以转换成NumPy数列的类型上执行。将含有一个极小的虚部的复数转换为实数时，可以考虑numpy.real_if_close函数。

有时编码时需要检查一个数字是不是标量（Python（长）整型， Python浮点型，Python复数或者rank-0数列）。numpy.isscalar函数提供该功能，返回值为0或1。

### Other useful functions

还应该提到其他一些有用的功能。对于进行相位处理，angle和unwrap函数非常有用。此外，linspace和logspace函数以线性或对数刻度返回等距的样本。最后，了解NumPy的索引编制功能很有用。应该提到函数select，它扩展了where在多个条件和多个选择时功能。调用约定为```select(condlist, choicelist, default = 0)```。 numpy.select是多个if语句的向量形式。它允许快速构造一个函数，该函数根据条件列表返回结果数列。返回数列的每个元素均取自choicelist，对应于condlist中为真的第一个条件。比如：

```python
>>> x = np.arange(10)
>>> condlist = [x<3, x>5]
>>> choicelist = [x, x**2]
>>> np.select(condlist, choicelist)
array([ 0,  1,  2,  0,  0,  0, 36, 49, 64, 81])
```

另外一些有用的函数也可以在scipy.special模块中找到，比如factorial和comb函数使用精确的整数算术（由于Python的Long整数对象） $n!$和$n!/k!(n-k)!$  ，或者使用浮点精度的gamma函数计算。

*以下这段不懂*

其他有用的函数可以在scipy.misc中找到。比如两个函数可用于使用离散差来求函数的近似导数。函数central_diff_weights返回等距N点近似o阶导数的加权系数。这些权重必须乘以与这些点相对应的函数，然后将结果相加以获得导数近似值。仅当该函数的样本可用时，才应使用此函数。当函数是可以传递给例程并进行估计的对象时，可以使用函数derivative在正确的点处自动评估该对象，以获得给定点处第o阶导数的N点近似值。

# Statistics (scipy.stats)

## Introduction

本章中，我们讨论了许多，但肯定不是所有的```scipy.stats```的功能。此处的目的是向用户提供此软件包的实用知识。有关更多详细信息，请参阅[参考手册](https://docs.scipy.org/doc/scipy/reference/stats.html#statsrefmanual)

注意，以下文档还在进行中：

- Discrete Statistical Distributions
- Continuous Statistical Distributions

## Random variables

有两种典型的分布类型，连续型随机变量和离散型随机变量。这些类中有超过80种连续型随机变量（Random Variables, RVs）和10种离散型随机变量，除此之外，用户可以很容易增加新的例程和分布。

所有的统计函数位于子包scipy.stats中，使用```info(stats)```命令可以得到相当完整的函数列表。可用的随机变量列表也可以通过stats子包的文档字符串获得。

以下主要讨论连续型RVs，几乎全部也都可以应用于离散型，同时列出一些差异（Specific points for discrete distributions）。

代码中，默认scipy.stats包以如下方式导入：

```python
>>> from scipy import stats
```

一些示例中，单独的对象导入方式：

```python
>>> from scipy.stats import norm
```

为了保持Python 2和Python 3的一致性，```print``是以下函数：

```python
>>> from __future__ import print_function
```

### Getting help

首先，所有的分布都有帮助功能。为了获取一些基本信息，打印相关的文档字符串：```print(stats.norm.__doc__)```。

为了获取帮助，比如，分布最高和最低限，调用：

```python
>>> print('bounds of distribution lower: %s, upper: %s' % (norm.a, norm.b))
bounds of distribution lower: -inf, upper: inf
```

可以使用```dir(norm)```列出分布的所有方法和性质。有些方法是私有的，虽然没有如此命名（名称没有以下划线起始），比如```veccdf```，只对内部计算可用。

为获取真正的主方法，我们列出冻结分布的方法。（以下会解释何为*冻结*分布）

```python
>>> rv = norm()
>>> dir(rv)  # reformatted
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'a', 'args', 'b', 'cdf', 'dist', 'entropy', 'expect', 'interval', 'isf', 'kwds', 'logcdf', 'logpdf', 'logpmf', 'logsf', 'mean', 'median', 'moment', 'pdf', 'pmf', 'ppf', 'random_state', 'rvs', 'sf', 'stats', 'std', 'support', 'var']
```

最后，我们可以通过反集获得可用分布的列表。

```python
>>> dist_continu = [d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]
>>> dist_discrete = [d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_discrete)]
>>> print('number of continuous distributions: %d' % len(dist_continu))
number of continuous distributions: 100
>>> print('number of discrete distributions:   %d' % len(dist_discrete))
number of discrete distributions:   15
```

### Common methods

连续型随机变量主要公共方法有：

- rvs：随机变量（Random Variates）
- pdf：概率密度函数（Probability Density Function）
- cdf：累积分布函数（Cumulative Distribution Function）
- sf：生存函数（Survival Function (1-CDF)）
- ppf：百分点函数（累积分布反函数）（Percent Point Function (Inverse of CDF)）
- isf：生存反函数（Inverse Survival Function (Inverse of SF)）
- stats：返回平均值，方差，Fisher偏度，Fisher峰度（mean, variance, (Fisher’s) skew, or (Fisher’s) kurtosis）
- moment：分布的非中心矩（non-central moments of the distribution）

让我们利用正态随机变量作为一个例子。

```python
>>> norm.cdf(0)
0.5
```

传入numpy数列可以计算每个点的数值的```cdf```：

```python
>>> norm.cdf([-1., 0, 1])
array([0.15865525, 0.5       , 0.84134475])
>>> import numpy as np
>>> norm.cdf(np.array([-1., 0, 1]))
array([0.15865525, 0.5       , 0.84134475])
```

所以，基本的方法，比如*pdf*， *cdf*等等，都是向量化的。

其他一般用到的方法也是如此：

```python
>>> norm.mean(), norm.std(), norm.var()
(0.0, 1.0, 1.0)
>>> norm.stats(moments="mv")
(array(0.), array(1.))
```

使用百分点函数```ppf```可以计算分布的中位数，这也是```cdf```的反函数：

```python
>>> norm.ppf(0.5)
0.0
```

使用```size```关键字可以生成随机变量序列：

```python
>>> norm.rvs(size=3)
array([-1.68701029, -0.16442867,  1.07258167])    #random
```

注意生成随机数据依赖于numpy.random包的生成器。上述例子中，特定的随机数流在各次运行之间不可重现。为了重现，需要明确全局变量种子

```python
>>> np.random.seed(1234)
```

虽然如此，不建议依赖全局的状态。一个更好的方法是使用*random_state*参数，它接受[numpy.random.mtrand.RandomState](https://docs.scipy.org/doc/numpy/reference/random/legacy.html#numpy.random.mtrand.RandomState)类的实例，或者一个整数，用来作为内部```RandomState```对象的种子：

```python
>>> norm.rvs(size=5, random_state=1234)
array([ 0.47143516, -1.19097569,  1.43270697, -0.3126519 , -0.72058873])
```

```norm.rev(5)```不会生成5个变量：

```python
>>> norm.rvs(5)
5.471435163732493
```

这里没有关键字的```5```解释为首个可能的关键字参数```loc```，这是所有连续分布都具有的一对关键字变量的第一个。是下一节的主题。

### Shifting and scaling

所有连续分布接受`loc`和`scale`作为关键字参数来调整分布的位置和范围，例如，对于标准正态分布，位置是平均值，范围是标准差。

```python
>>> norm.stats(loc=3, scale=4, moments="mv")
(array(3.), array(16.))
```

很多情况下，随机变量`X`分布可以通过$(X-loc)/scale$ 进行标准化。默认值$loc=0$，$scale=1$。

使用好`loc`和`scale`参数可以将标准分布修改为很多种形式。为了深入的介绍数据缩放，给出一个平均值为$1/\lambda$的几何分布随机变量
$$
F(x)=1-exp(-\lambda x)
$$
通过使用所述缩放规则，使`scale = 1./lambda`可以看到得到了合适的范围。

```python
>>> from scipy.stats import expon
>>> expon.mean(scale=3.)
3.0
```

> 注意：
>
> 采用形状参数的分布想达到需要的形式，需要的可能不止是`loc`和`scale`。比如，给定长度为R的特定2维向量的长度受到每个组分独立$N(0, /\sigma^2)$偏差影响的分布是$rice(R/\sigma, scale=\sigma)$， 第一个形状参数需要与$x$一起缩放。 

均匀分布也很有趣：

```python
>>> from scipy.stats import uniform
>>> uniform.cdf([0, 1, 2, 3, 4, 5], loc=1, scale=4)
array([0.  , 0.  , 0.25, 0.5 , 0.75, 1.
```

最后，说回前面遗留的问题`norm.rvs(5)`意义，实际上是生成一个分布，第一个参数5，传给了`loc`参数：

```python
>>> np.mean(norm.rvs(5, size=500))
5.009835510696999
```

因此，解释一下`norm.rvs(5)`的输出，它生成了一个正态分布随机变量，平均值是`loc=5`，因为默认的`size=1`。

建议通过关键字明确的设定`loc`和`scale`参数，当对随机变量使用冻结分布调用多个方法里，可以减少重复次数。

### Shape parameters

一般的连续随机变量可以通过`loc`和`scale`参数进行改变和缩放，但一些分布需要额外的形状参数。比如，带有密度的伽玛分布
$$
\gamma(x, a)=\frac{\lambda(\lambda x)^{a-1}}{\Gamma(a)} e^{-\lambda x}
$$
要求形状参数$a$，可通过将`scale`关键字设定为$1/\lambda$来设置$\lambda$。

来看一下伽玛分布的形状参数的数值和名称（通过上述可知为1）.

```python
>>> from scipy.stats import gamma
>>> gamma.numargs
1
>>> gamma.shapes
'a'
```

将形状变量设置为1得到指数分布，所以可以轻松比较是否得到了想要的结果。

```python
>>> gamma(1, scale=2.).stats(moments="mv")
(array(2.), array(4.))
```

注意到也可以通过形状参数关键字设置：

```python
>>> gamma(a=1, scale=2.).stats(moments="mv")
(array(2.), array(4.))
```

### Freezing a distribution

每次都传入`loc`和`scale`参数会很麻烦，*冻结*随机变量的概念就是要解决这种问题。

```python
>>> rv = gamma(1, scale=2)
```

通过使用`rv`就不再需要包括缩放和形状参数。因此，可以通过这两种方式使用分布，要么将所有的分布参数传入每个调用的方法（像之前做的一样），或者创建分布里即冻结参数。比如：

```python
>>> rv.mean(), rv.std()
(2.0, 2.0)
```

### Broadcasting

`pdf`等基本方法满足numpy的通用广播规则。比如，可以计算t分布不同概率和自由度下右尾的临界值。

```python
>>> stats.t.isf([0.1, 0.05, 0.01], [[10], [11]])
array([[1.37218364, 1.81246112, 2.76376946],
       [1.36343032, 1.79588482, 2.71807918]])
```

这里，第一行包含自由度为10的临界值，第二行为自由度为11的临界值，所以广播规则可以给出需要调用`isf`两次的结果：

```python
>>> stats.t.isf([0.1, 0.05, 0.01], 10)
array([1.37218364, 1.81246112, 2.76376946])
>>> stats.t.isf([0.1, 0.05, 0.01], 11)
array([1.36343032, 1.79588482, 2.71807918])
```

如果表示概率的数列，即`[0.1, 0.05, 0.01]`和自由度的数列，即`[[10], [11]]`有相同的数列形状，会进行逐元素的匹配。举个例子，可能通过以下方式获取自由度分别为10，11和12并对应尾部概率为10%，5%和1%的临界值

```python
>>> stats.t.isf([0.1, 0.05, 0.01], [10, 11, 12])
array([1.37218364, 1.79588482, 2.68099799])
```

### Specific points for discrete distributions

离散分布具有大部分与连续分布相同的基本方法。不过，`pdf`被概率质量函数`pmf`取代，没有估计方法可用，比如fit，`scale`也不是一个有效的关键字参数，位置参数，关键字`loc`还是可以用来改变分布。

cdf的计算需要额外注意。在连续分布中，累积分布函数在大部分情况下区间内(a,b)单调递增，因此有唯一反函数。离散分布的cdf是一个阶梯函数，所以cdf的反函数，即百分点函数需要不同的定义：

```
ppf(q) = min{x : cdf(x) >= q, x integer}
```

[点击此处了解详细信息](https://docs.scipy.org/doc/scipy/reference/tutorial/stats/discrete.html#percent-point-function-inverse-cdf)

利用超几何分布进行示例

```python
>>> from scipy.stats import hypergeom
>>> [M, n, N] = [20, 7, 12]
```

如果在某些整数点使用cdf，在这些cdf值处估计ppf，可以得到开始的整数值，比如

```python
>>> x = np.arange(4)*2
>>> x
array([0, 2, 4, 6])
>>> prb = hypergeom.cdf(x, M, n, N)
>>> prb
array([1.03199174e-04, 5.21155831e-02, 6.08359133e-01, 9.89783282e-01])
>>> hypergeom.ppf(prb, M, n, N)
array([0., 2., 4., 6.])
```

如果使用的值不在cdf阶梯函数的联结处，我们将得到下一个更高的整数。

### Fitting distributions

非冻结分布的附加方法与分布的参数估计有关：

- fit: 分布参数的最大似然估计，包括位置和范围
- fit_loc_scale: 形状参数给定的情况下对位置和范围的估计
- nnlf: 负对数似然函数
- expect: 根据pdf或pmf计算函数的期望值

### Performance issues and cautionary remarks

每个方法在速度方面的性能，不同的分布和方法差异很大。两种方式可能得到某方法的结果：或者是精确的计算，或者是通过独立于特定分布的通用算法。

一方面，准确的计算，需要直接定义给定分布的方法，通过解析公式，或者随机变量的`scipy.special`或`numpy.random`特殊函数。这些通常相对比较快。

另一方面，通用方法用于分布或者计算公式不明确情况下。定义一个分布只需要pdf或者cdf其中之一； 其他方法可以通过数值积分和根查找推断。然而这些间接的方法会很慢。举例来说，`rgh = stats.gausshyper.rvs(0.5, 2, 2, 2, size=100)`间接的创建100个随机变量，大约花费了19秒的时间，然而100万个标准正态分布或者t分布的随机变量只需要大约1秒钟。

### Remaining issues

`scipy.stats`中的分布已经完善了很多，并且有相当多检验包。然而还存在一些问题：

- 这些分布已经在一些参数范围内进行了测试；但是，在一些范围内，可能仍然存在一些不正确的结果。
- *fit*中的最大似然估计默认启动参数不适用于所有分布，用户需要提供良好的启动参数。此外，对于某些分布，使用最大似然估计量可能不是最佳选择。

## Building specific distributions

下一个示例展示如何建立自己的分布，更多的示例展示了分布的用法和一些统计检验。

### Making a continuous distribution, i.e., subclassing `rv_continuous`

建立一个连续分布比较简单。

```python
>>> from scipy import stats
>>> class deterministic_gen(stats.rv_continuous):
...     def _cdf(self, x):
...             return np.where(x < 0, 0., 1.)
...     def _stats(self):
...             return 0., 0., 0., 0.
```

```python
>>> deterministic = deterministic_gen(name="deterministic")
>>> deterministic.cdf(np.arange(-3, 3, 0.5))
array([0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.])
```

有趣的是，`pdf`可以自动计算：

```python
>>> deterministic.pdf(np.arange(-3, 3, 0.5))
array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 5.83333333e+04, 4.16333634e-12,
       4.16333634e-12, 4.16333634e-12, 4.16333634e-12, 4.16333634e-12])
```

注意Performance issues and cautionary remarks部分讲的性能问题。未指定常用方法的计算可能会非常慢，因为仅调用了通用方法，从本质上讲，它们无法使用有关分布的任何特定信息。作为警告示例：

```python
>>> from scipy.integrate import quad
>>> quad(deterministic.pdf, -1e-1, 1e-1)
(4.163336342344337e-13, 0.0)
```

这个结果是不对的：pdf的积分应该是1。积分区间再小一点：

```python
>>> quad(deterministic.pdf, -1e-3, 1e-3)
__main__:1: IntegrationWarning: The maximum number of subdivisions (50) has been achieved.
  If increasing the limit yields no improvement it is advised to analyze
  the integrand in order to determine the difficulties.  If the position of a
  local difficulty can be determined (singularity, discontinuity) one will
  probably gain from splitting up the interval and calling the integrator
  on the subranges.  Perhaps a special-purpose integrator should be used.
(1.000076872229173, 0.0010625571718182458)
```

这样看起来好多了。产生问题的原因是由于pdf在deterministic分布的类中没有定义的事实。

### Subclassing `rv_discrete`

下面我们使用`stats.rv_discrete`生成一个离散分布，该分布具有以整数为中心区间的截断正态概率。

**General info**

来自于rv_discrete的文档字符串，`help(stats.rv_discrete)`

> 通过传递给rv_discrete初始化方法（values=keyword）可以构建任何离散随机变量，使得P{X=xk}=pk，序列元组(xk, pk)描述了X的取值(xk)发生的非零概率(pk)。

在这之后，有完成该工作更多的要求：

- 关键字*name*必需
- 分布中的点xk必须是整数
- 需要指定有效位数（小数）

如果后面两个条件不满足，会引发异常或结果不正确。

**An example**

开始工作，首先：

```python
>>> npoints = 20   # 分布的整数点个数减 1
>>> npointsh = npoints // 2
>>> npointsf = float(npoints)
>>> nbound = 4   # 截断正态分布的边界
>>> normbound = (1+1/npointsf) * nbound   # 截断正态分布的实际边界
>>> grid = np.arange(-npointsh, npointsh+2, 1)   # 整数网格
>>> gridlimitsnorm = (grid-0.5) / npointsh * nbound   # 截断正态分布的步长限制
>>> gridlimits = grid - 0.5   # 分析时使用后者
>>> grid = grid[:-1]
>>> probs = np.diff(stats.truncnorm.cdf(gridlimitsnorm, -normbound, normbound))
>>> gridint = grid
```

最后可以使用子类`rv_discrete`：

```python
>>> normdiscrete = stats.rv_discrete(values=(gridint, np.round(probs, decimals=7)), name='normdiscrete')
```

现在已经定义了分布，可以访问离散分布的所有通用方法。

```python
>>> print('mean = %6.4f, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f' % normdiscrete.stats(moments='mvsk'))
mean = -0.0000, variance = 6.3302, skew = 0.0000, kurtosis = -0.0076
```

```python
>>> nd_std = np.sqrt(normdiscrete.stats(moments='v'))
```

**Testing the implementation**

生成一个随机样本并比较观察到的频率与概率。

```python
>>> n_sample = 500
>>> np.random.seed(87655678)
>>> rvs = normdiscrete.rvs(size=n_sample)
>>> f, l = np.histogram(rvs, bins=gridlimits)
>>> sfreq = np.vstack([gridint, f, probs*n_sample]).T
>>> print(sfreq)
[[-1.00000000e+01  0.00000000e+00  2.95019349e-02]
 [-9.00000000e+00  0.00000000e+00  1.32294142e-01]
 [-8.00000000e+00  0.00000000e+00  5.06497902e-01]
 [-7.00000000e+00  2.00000000e+00  1.65568919e+00]
 [-6.00000000e+00  1.00000000e+00  4.62125309e+00]
 [-5.00000000e+00  9.00000000e+00  1.10137298e+01]
 [-4.00000000e+00  2.60000000e+01  2.24137683e+01]
 [-3.00000000e+00  3.70000000e+01  3.89503370e+01]
 [-2.00000000e+00  5.10000000e+01  5.78004747e+01]
 [-1.00000000e+00  7.10000000e+01  7.32455414e+01]
 [ 0.00000000e+00  7.40000000e+01  7.92618251e+01]
 [ 1.00000000e+00  8.90000000e+01  7.32455414e+01]
 [ 2.00000000e+00  5.50000000e+01  5.78004747e+01]
 [ 3.00000000e+00  5.00000000e+01  3.89503370e+01]
 [ 4.00000000e+00  1.70000000e+01  2.24137683e+01]
 [ 5.00000000e+00  1.10000000e+01  1.10137298e+01]
 [ 6.00000000e+00  4.00000000e+00  4.62125309e+00]
 [ 7.00000000e+00  3.00000000e+00  1.65568919e+00]
 [ 8.00000000e+00  0.00000000e+00  5.06497902e-01]
 [ 9.00000000e+00  0.00000000e+00  1.32294142e-01]
 [ 1.00000000e+01  0.00000000e+00  2.95019349e-02]]
```

![](https://docs.scipy.org/doc/scipy/reference/_images/normdiscr_plot1.png)

![../_images/normdiscr_plot2.png](https://docs.scipy.org/doc/scipy/reference/_images/normdiscr_plot2.png)

接下来，测试样本是否由norm-discrete分布产生。这也可以通过随机数字是否正常来验证。

卡方检验要求每个步长中最小观察数量。将尾部的步长合并成大的步长使得每个步长包含足够的观察数量。

```python
>>> f2 = np.hstack([f[:5].sum(), f[5:-5], f[-5:].sum()])
>>> p2 = np.hstack([probs[:5].sum(), probs[5:-5], probs[-5:].sum()])
```

```python
>>> ch2, pval = stats.chisquare(f2, p2*n_sample)
>>> print('chisquare for normdiscrete: chi2 = %6.3f pvalue = %6.4f' % (ch2, pval))
chisquare for normdiscrete: chi2 = 12.466 pvalue = 0.4090
```

本例中p值很高，所以我们可以比较确信随机样本由分布产生。

## Analysing one sample

首先创建一些随机变量。设定种子使得每次运行可以得到相同的结果。作为例子，由Student t分布中取一个样本：

```python
>>> np.random.seed(282629734)
>>> x = stats.t.rvs(10, size=1000)
```

这里设置了t分布需要的形状参数自由度为10。size=1000表示样本包含1000个独立绘制的（伪）随机数字。由于没有明确指定，关键字参数*loc*和*scale*为默认的0和1。

### Descriptive statistics

*x*是numpy数列，可以直接访问所有的数列方法，即：

```python
>>> print(x.min())      #  与np.min(x)相同
-3.7897557242248197
>>> print(x.max())      #  与np.max(x)相同
5.263277329807165
>>> print(x.mean())      #  与np.mean(x)相同
0.014061066398468422
>>> print(x.var())      #  与np.var(x)相同
1.288993862079285
```

样本的性质与理论值比较如何？

```python
>>> m, v, s, k = stats.t.stats(10, moments='mvsk')
>>> n, (smin, smax), sm, sv, ss, sk = stats.describe(x)
```

```python
>>> sstr = '%-14s mean = %6.4f, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f'
>>> print(sstr % ('distribution:', m, v, s ,k))
distribution:  mean = 0.0000, variance = 1.2500, skew = 0.0000, kurtosis = 1.0000
>>> print(sstr % ('sample:', sm, sv, ss, sk))
sample:        mean = 0.0141, variance = 1.2903, skew = 0.2165, kurtosis = 1.0556
```

注意：对于方差，stats.describe使用无偏估计，而np.var是有偏估计。

这个样本的统计值 与理论值相比有微量的差别。

### T-test and KS-test

可以使用t检验来检验样本与理论的平均值是否有统计的显著差异。

```python
>>> print('t-statistic = %6.3f pvalue = %6.4f' %  stats.ttest_1samp(x, m))
t-statistic =  0.391 pvalue = 0.6955
```

p值是0.7，这意味着在指定的alpha错误下，比如10%，不能拒绝样本平均值为0，即等于标准t分布的期望值的假设。

作为练习，可以不使用提供的函数，直接计算t检验值，会得到相同的结果：

```python
>>> tt = (sm-m)/np.sqrt(sv/float(n))   # 平均值的t值
>>> pval = stats.t.sf(np.abs(tt), n-1)*2  # 双端p值 = (abs(t)>tt)的概率
>>> print('t-statistic = %6.3f pvalue = %6.4f' % (tt, pval))
t-statistic =  0.391 pvalue = 0.6955
```

Kolmogorov-Smirnov(K-S)检验用来检验样本来源于标准t分布的假设

```python
>>> print('KS-statistic D = %6.3f pvalue = %6.4f' % stats.kstest(x, 't', (10,)))
KS-statistic D =  0.016 pvalue = 0.9606
```

p值足够高，所以不可以拒绝随机样本符合t分布的假设。真实的应用中，并不知道是什么分布，如果对样本对标准正态分布上做K-S检验，也不可以拒绝样本由正态分布产生的假设。本例中，p值接近40%。

```python
>>> print('KS-statistic D = %6.3f pvalue = %6.4f' % stats.kstest(x, 'norm'))
KS-statistic D =  0.028 pvalue = 0.3949
```

然而，标准正态分布方差为1，本样本方差是1.29。如果标准化样本并且与正态分布作检验，p值仍然大到不能拒绝样本来源于正态分布的假设。

```python
>>> d, pval = stats.kstest((x-x.mean())/x.std(), 'norm')
>>> print('KS-statistic D = %6.3f pvalue = %6.4f' % (d, pval))
KS-statistic D =  0.032 pvalue = 0.2402
```

注意：Kolmogorov-Smirnov检验假设针对给定参数的分布进行检验，因此最后，平均值和方差的估计违反了该假设，分布的检验统计和基于此得到的检验p值也是不正确的。

### Tails of the distribution

最后，可以检查分布的右尾。使用百分点函数ppf得到临界值，这是cdf的反函数，或者更直接的使用生存函数的反函数。

```python
>>> crit01, crit05, crit10 = stats.t.ppf([1-0.01, 1-0.05, 1-0.1], 10)
>>> print('critical values from ppf at 1%%, 5%% and 10%% %8.4f %8.4f %8.4f' % (crit01, crit05, crit10))
critical values from ppf at 1%, 5% and 10%   2.7638   1.8125   1.3722
>>> print('critical values from isf at 1%%, 5%% and 10%% %8.4f %8.4f %8.4f' % tuple(stats.t.isf([0.01,0.05,0.10],10)))
critical values from isf at 1%, 5% and 10%   2.7638   1.8125   1.3722
```

```python
>>> freq01 = np.sum(x>crit01) / float(n) * 100
>>> freq05 = np.sum(x>crit05) / float(n) * 100
>>> freq10 = np.sum(x>crit10) / float(n) * 100
>>> print('sample %%-frequency at 1%%, 5%% and 10%% tail %8.4f %8.4f %8.4f' % (freq01, freq05, freq10))
sample %-frequency at 1%, 5% and 10% tail   1.4000   5.8000  10.5000
```

三种情况中，样本比总体的分布右尾有更高的权重。可以简单的增加样本量看是否得到更接近的结果。这种，样本上的频率与理论概率应该更接近，但是如果重复几次，波动依然很大。

```python
>>> freq05l = np.sum(stats.t.rvs(10, size=10000) > crit05) / 10000.0 * 100
>>> print('larger sample %%-frequency at 5%% tail %8.4f' % freq05l)
larger sample %-frequency at 5% tail   4.8000
```

也可以与正态分布的尾部进行比较，正态分布的尾部权重更小：

```python
>>> print('tail prob. of normal at 1%%, 5%% and 10%% %8.4f %8.4f %8.4f' % tuple(stats.norm.sf([crit01, crit05, crit10])*100))
tail prob. of normal at 1%, 5% and 10%   0.2857   3.4957   8.5003
```

对于有限个数的步长，卡方检验可以用来检验观察频率是否与假设的分布概率有差异。

```python
>>> quantiles = [0.0, 0.01, 0.05, 0.1, 1-0.10, 1-0.05, 1-0.01, 1.0]
>>> crit = stats.t.ppf(quantiles, 10)
>>> crit
array([       -inf, -2.76376946, -1.81246112, -1.37218364,  1.37218364,
        1.81246112,  2.76376946,         inf])
>>> n_sample = x.size
>>> freqcount = np.histogram(x, bins=crit)[0]
>>> tprob = np.diff(quantiles)
>>> nprob = np.diff(stats.norm.cdf(crit))
>>> tch, tpval = stats.chisquare(freqcount, tprob*n_sample)
>>> nch, npval = stats.chisquare(freqcount, nprob*n_sample)
>>> print('chisquare for t: chi2 = %6.2f pvalue = %6.4f' % (tch, tpval))
chisquare for t: chi2 =   2.30 pvalue = 0.8901
>>> print('chisquare for normal: chi2 = %6.2f pvalue = %6.4f' %(nch, npval))
chisquare for normal: chi2 =  64.60 pvalue = 0.0000
```

可以看到标准正态分布明确的被拒绝了，而标准t分布不能被拒绝。由于样本方差与两个标准分布都不同，所以可以考虑缩放和位置的估计后再做一次。

分布的fit方法可以用来估计分布的参数，使用估计分布的概率再进行一次检验。

```python
>>> tdof, tloc, tscale = stats.t.fit(x)
>>> nloc, nscale = stats.norm.fit(x)
>>> tprob = np.diff(stats.t.cdf(crit, tdof, loc=tloc, scale=tscale))
>>> nprob = np.diff(stats.norm.cdf(crit, loc=nloc, scale=nscale))
>>> tch, tpval = stats.chisquare(freqcount, tprob*n_sample)
>>> nch, npval = stats.chisquare(freqcount, nprob*n_sample)
>>> print('chisquare for t: chi2 = %6.2f pvalue = %6.4f' % (tch, tpval))
chisquare for t: chi2 =   1.58 pvalue = 0.9542
>>> print('chisquare for normal: chi2 = %6.2f pvalue = %6.4f' % (nch, npval))
chisquare for normal: chi2 =  11.08 pvalue = 0.0858
```

考虑了估计的参数，仍然拒绝样本来自于正态分布的假设（5%水平），但是，不能拒绝t分布，因为p值是0.95。

### Special tests for normal distributions

由于正态分布是统计中最常见的分布，有一些额外的函数可以进行检验一个样本是否来自于正态分布。

首先可以检验样本的偏度和峰度是否与一个正态分布有差异：

```python
>>> print('normal skewtest teststat = %6.3f pvalue = %6.4f' % stats.skewtest(x))
normal skewtest teststat =  2.785 pvalue = 0.0054
>>> print('normal kurtosistest teststat = %6.3f pvalue = %6.4f' % stats.kurtosistest(x))
normal kurtosistest teststat =  4.757 pvalue = 0.0000
```

正态检验将两者结合起来

```python
>>> print('normaltests teststat = %6.3f pvalue = %6.4f' % stats.normaltest(x))
normaltests teststat = 30.379 pvalue = 0.0000
```

三个检验中p值都很低，可以拒绝样本具有正态分布的偏度和峰度的假设。

由于样本的偏度和峰度是基于中心矩的，如果检验标准化的样本，也可以得到完全相同的结果：

```python
>>> print('normaltest teststat = %6.3f pvalue = %6.4f' % stats.normaltest((x-x.mean())/x.std()))
normaltest teststat = 30.379 pvalue = 0.0000
```

由于如此强烈的拒绝了正态性，可以测试正态检验在另外一个例子中是否给出合理的结果：

（译者注：取决随机数种子设定，以下代码运行可能会产生不同结果）

```python
>>> print('normaltest teststat = %6.3f pvalue = %6.4f' %
...       stats.normaltest(stats.t.rvs(10, size=100)))
normaltest teststat =  4.698 pvalue = 0.0955
>>> print('normaltest teststat = %6.3f pvalue = %6.4f' %
...              stats.normaltest(stats.norm.rvs(size=1000)))
normaltest teststat =  0.613 pvalue = 0.7361
```

当检验t分布的小样本或正态分布的大样本的正态性时，都不能拒绝原假设，样本来源于正态分布总体。第一个例子中，是因为小样本情况下检验没有能力区分t分布和正态分布的随机变量。

## Comparing two samples

以下给出两个样本，可以来自相同或者不同的分布，检验两个样本是否有相同的统计值。

### Comparing means

检验具有相同平均值的样本：

```python
>>> rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
>>> rvs2 = stats.norm.rvs(loc=5, scale=10, size=500)
>>> stats.ttest_ind(rvs1, rvs2)
Ttest_indResult(statistic=-0.40108918347419803, pvalue=0.6884403219876052)
```

检验具有不同平均值的样本：

```python
>>> rvs3 = stats.norm.rvs(loc=8, scale=10, size=500)
>>> stats.ttest_ind(rvs1, rvs3)
Ttest_indResult(statistic=-4.296151964438256, pvalue=1.9078085541582206e-05)
```

### Kolmogorov-Smirnov test for two samples ks_2samp

举个例子，两个样本来自相同分布，由于p值比较高，不能拒绝原假设

```python
>>> stats.ks_2samp(rvs1, rvs2)
Ks_2sampResult(statistic=0.032, pvalue=0.9603008958861495)
```

第二个例子，具有不同的位置，即平均值，p值小于1%，可以拒绝原假设

```python
>>> stats.ks_2samp(rvs1, rvs3)
Ks_2sampResult(statistic=0.128, pvalue=0.0005458774578140435)
```

## Kernel density estimation

统计中常见的任务是从一组数据样本的随机变量中估计概率密度函数( probability density function, PDF)，这叫做密度估计。最广泛工具是直方图。直方图是可视化很有用的工具（主要是所有人都理解），但是数据利用效率不高。核密度估计(Kernel density estimation, KDE)是解决同样任务更有效的工具。高斯核密度估计(gaussian_kde)可以用来估计单变量或者多变量数据的概率密度函数，如果数据是单峰的话效果更好。

### Univariate estimation

先用一个小数据来看看高斯核密度估计的用法及bandwidth的不同选项。数据样本的概率密度函数在图中由底部的蓝色"+”表示（rug图）：

```python
>>> x1 = np.array([-7, -5, -1, 4, 5], dtype=np.float)
>>> kde1 = stats.gaussian_kde(x1)
>>> kde2 = stats.gaussian_kde(x1, bw_method='silverman')
```

```python
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
```

```python
>>> ax.plot(x1, np.zeros(x1.shape), 'b+', ms=20)   # rug plot
[<matplotlib.lines.Line2D object at 0x1a1eb80b90>]
>>> x_eval = np.linspace(-10, 10, num=200)
>>> ax.plot(x_eval, kde1(x_eval), 'k-', label="Scott's Rule")
[<matplotlib.lines.Line2D object at 0x1a234ca450>]
>>> ax.plot(x_eval, kde2(x_eval), 'r-', label="Silverman's Rule")
[<matplotlib.lines.Line2D object at 0x1a234ca510>]
```

```python
>>> plt.show()
```

![](https://docs.scipy.org/doc/scipy/reference/_images/stats-1.png)

可以看到Scott's和Silverman's规则有些不同，这么少的数据bandwidth选择的太宽了。可以定义bandwidth函数得到更不平滑的结果。

```python
>>> def my_kde_bandwidth(obj, fac=1./5):
...     """使用Scott's规则， 乘以连续的因子"""
...     return np.power(obj.n, -1./(obj.d+4)) * fac
```

```python
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
```

```python
>>> ax.plot(x1, np.zeros(x1.shape), 'b+', ms=20)    # rug plot
[<matplotlib.lines.Line2D object at 0x1a27948690>]
>>> kde3 = stats.gaussian_kde(x1, bw_method=my_kde_bandwidth)
>>> ax.plot(x_eval, kde3(x_eval), 'g-', label="With smaller BW")
[<matplotlib.lines.Line2D object at 0x1a27948d50>]
```

```python
>>> plt.show()
```

![](https://docs.scipy.org/doc/scipy/reference/_images/kde_plot2.png)

看到如果将bandwidth设置的很窄，获得的概率密度函数估计仅仅是每个数据点的高斯和。

现在使用更实际的例子看一下两种bandwidth选择规则的不同。众所周知，这些规则对于（接近）正态分布非常有效，但是即使对于非常强的非正态分布的单峰分布，它们也相当有效。使用自由度为5的Student's T分布作为非正态分布例子。

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from scipy import stats
>>>
>>> np.random.seed(12456)
>>> x1 = np.random.normal(size=200)  # random data, normal distribution
>>> xs = np.linspace(x1.min()-1, x1.max()+1, 200)
>>> kde1 = stats.gaussian_kde(x1)
>>> kde2 = stats.gaussian_kde(x1, bw_method='silverman')
>>> fig = plt.figure(figsize=(8, 6))
>>> ax1 = fig.add_subplot(211)
>>> ax1.plot(x1, np.zeros(x1.shape), 'b+', ms=12)  # rug plot
[<matplotlib.lines.Line2D object at 0x1a23a42ad0>]
>>> ax1.plot(xs, kde1(xs), 'k-', label="Scott's Rule")
[<matplotlib.lines.Line2D object at 0x1a23a42f90>]
>>> ax1.plot(xs, kde2(xs), 'b-', label="Silverman's Rule")
[<matplotlib.lines.Line2D object at 0x1a23a50650>]
>>> ax1.plot(xs, stats.norm.pdf(xs), 'r--', label="True PDF")
[<matplotlib.lines.Line2D object at 0x1a23a50b90>]
>>> ax1.set_xlabel('x')
Text(0.5, 0, 'x')
>>> ax1.set_ylabel('Density')
Text(0, 0.5, 'Density')
>>> ax1.set_title("Normal (top) and Student's T$_{df=5}$ (bottom) distributions")
Text(0.5, 1.0, "Normal (top) and Student's T$_{df=5}$ (bottom) distributions")
>>> ax1.legend(loc=1)
<matplotlib.legend.Legend object at 0x1a234b0cd0>
>>> x2 = stats.t.rvs(5, size=200)  # random data, T distribution
>>> xs = np.linspace(x2.min() - 1, x2.max() + 1, 200)
>>> kde3 = stats.gaussian_kde(x2)
>>> kde4 = stats.gaussian_kde(x2, bw_method='silverman')
>>> ax2 = fig.add_subplot(212)
>>> ax2.plot(x2, np.zeros(x2.shape), 'b+', ms=12)  # rug plot
[<matplotlib.lines.Line2D object at 0x1a23c23f50>]
>>> ax2.plot(xs, kde3(xs), 'k-', label="Scott's Rule")
[<matplotlib.lines.Line2D object at 0x1a23a61b50>]
>>> ax2.plot(xs, kde4(xs), 'b-', label="Silverman's Rule")
[<matplotlib.lines.Line2D object at 0x1a23a61a90>]
>>> ax2.plot(xs, stats.t.pdf(xs, 5), 'r--', label="True PDF")
[<matplotlib.lines.Line2D object at 0x1a23c32fd0>]
>>> ax2.set_xlabel('x')
Text(0.5, 0, 'x')
>>> ax2.set_ylabel('Density')
Text(0, 0.5, 'Density')
>>> plt.show()
```

![](https://docs.scipy.org/doc/scipy/reference/_images/kde_plot3.png)

现在我们来看看具有一个较宽和一个较窄的高斯特征的双峰分布。由于准确解析每个功能所需的bandwidths不同，我们预计这将是一个更难估计的密度。

```python
>>> from functools import partial
>>> loc1, scale1, size1 = (-2, 1, 175)
>>> loc2, scale2, size2 = (2, 0.2, 50)
>>> x2 = np.concatenate([np.random.normal(loc=loc1, scale=scale1, size=size1),
...                      np.random.normal(loc=loc2, scale=scale2, size=size2)])
>>> x_eval = np.linspace(x2.min() - 1, x2.max() + 1, 500)
>>> kde = stats.gaussian_kde(x2)
>>> kde2 = stats.gaussian_kde(x2, bw_method='silverman')
>>> kde3 = stats.gaussian_kde(x2, bw_method=partial(my_kde_bandwidth, fac=0.2))
>>> kde4 = stats.gaussian_kde(x2, bw_method=partial(my_kde_bandwidth, fac=0.5))
>>> pdf = stats.norm.pdf
>>> bimodal_pdf = pdf(x_eval, loc=loc1, scale=scale1) * float(size1) / x2.size + \
...               pdf(x_eval, loc=loc2, scale=scale2) * float(size2) / x2.size
>>> fig = plt.figure(figsize=(8, 6))
>>> ax = fig.add_subplot(111)
>>> ax.plot(x2, np.zeros(x2.shape), 'b+', ms=12)
>>> ax.plot(x_eval, kde(x_eval), 'k-', label="Scott's Rule")
>>> ax.plot(x_eval, kde2(x_eval), 'b-', label="Silverman's Rule")
>>> ax.plot(x_eval, kde3(x_eval), 'g-', label="Scott * 0.2")
>>> ax.plot(x_eval, kde4(x_eval), 'c-', label="Scott * 0.5")
>>> ax.plot(x_eval, bimodal_pdf, 'r--', label="Actual PDF")
>>> ax.set_xlim([x_eval.min(), x_eval.max()])
>>> ax.legend(loc=2)
>>> ax.set_xlabel('x')
>>> ax.set_ylabel('Density')
>>> plt.show()
```

![](https://docs.scipy.org/doc/scipy/reference/_images/kde_plot4.png)

不出所料，由于双峰分布的两个特征的特征尺寸不同，KDE并不像我们想要的那样接近真实的PDF。通过将默认bandwidths减半（Scott * 0.5），我们可以做得更好，不过使用比默认小5倍的bandwidths还不够平滑。但是在这种情况下，我们真正需要的是非均匀（自适应）bandwidths。

### Multivariate estimation

使用gaussian_kde可以像单变量一样进行多变量估计，下面用两个变量来说明。首先使用一个两个相关变量相关的模型产生一些随机变量。

```python
>>> def measure(n):
...     """测量模型，返回两个成对测量值"""
...     m1 = np.random.normal(size=n)
...     m2 = np.random.normal(scale=0.5, size=n)
...     return m1+m2, m1-m2
```

```python
>>> m1, m2 = measure(2000)
>>> xmin = m1.min()
>>> xmax = m1.max()
>>> ymin = m2.min()
>>> ymax = m2.max()
```

然后使用在数据上应用核密度估计（KDE）：

```python
>>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
>>> positions = np.vstack([X.ravel(), Y.ravel()])
>>> values = np.vstack([m1, m2])
>>> kernel = stats.gaussian_kde(values)
>>> z = np.reshape(kernel.evaluate(positions).T, X.shape)
```

最后在用颜色图标注出双变量的分布估计，并画出各个数据点。

```python
>>> fig = plt.figure(figsize=(8, 6))
>>> ax = fig.add_subplot(111)
>>> ax.imshow(np.rot90(z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
<matplotlib.image.AxesImage object at 0x1a23d623d0>
>>> ax.plot(m1, m2, 'k.', markersize=2)
[<matplotlib.lines.Line2D object at 0x1a23d62890>]
>>> ax.set_xlim([xmin, xmax])
(-4.673087018557033, 3.9997057527518556)
>>> ax.set_ylim([ymin, ymax])
(-3.9234039912730623, 3.310000395631509)
>>> plt.show()
```

![](https://docs.scipy.org/doc/scipy/reference/_images/kde_plot5.png)

### Multiscale Graph Correlation (MGC)

使用multiscale_graphcorr可以对高维和非线性的数据做独立检验。在开始前导入一些包：

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt; plt.style.use('classic')
>>> from scipy.stats import multiscale_graphcorr
```

使用自定义的plotting函数展示数据的关系：

```python
>>> def mgc_plot(x, y, sim_name, mgc_dict=None, only_viz=False, only_mgc=False):
...     """画sim和MGC图"""
...     if not only_mgc:
...             # 模拟
...             plt.figure(figsize=(8, 8))
...             ax = plt.gca()
...             ax.set_title(sim_name + "Simulation", fontsize=20)
...             ax.scatter(x, y)
...             ax.set_xlabel('X', fontsize=15)
...             ax.set_ylabel('Y', fontsize=15)
...             ax.axis('equal')
...             ax.tick_params(axis="x", labelsize=15)
...             ax.tick_params(axis="y", labelsize=15)
...             plt.show()
...     if not only_viz:
...             # 本地关系图
...             plt.figure(figsize=(8, 8))
...             ax = plt.gca()
...             mgc_map = mgc_dict["mgc_map"]
...             # 画热图
...             ax.set_title("Local Correlation Map", fontsize=20)
...             im = ax.imshow(mgc_map, cmap='YlGnBu')
...             # 颜色柱子
...             cbar = ax.figure.colorbar(im, ax=ax)
...             cbar.ax.set_ylabel("", rotation=-90, va="bottom")
...             ax.invert_yaxis()
...             # 关掉背景建立白色网格
...             for edge, spine in ax.spines.items():
...                     spine.set_visible(False)
...             # 比例优化
...             opt_scale = mgc_dict["opt_scale"]
...             ax.scatter(opt_scale[0], opt_scale[1], marker='X', s=200, color='red')
...             # 其他格式
...             ax.tick_params(bottom="off", left="off")
...             ax.set_xlabel('#Neighbors for X', fontsize=15)
...             ax.tick_params(axis="x", labelsize=15)
...             ax.tick_params(axis="y", labelsize=15)
...             ax.set_xlim(0, 100)
...             ax.set_ylim(0, 100)
...             plt.show()
```

先看一些线性数据：

```python
>>> np.random.seed(12345678)
>>> x = np.linspace(-1, 1, num=100)
>>> y = x + 0.3 * np.random.random(x.size)
```

模拟关系作图如下：

```python
>>> mgc_plot(x, y, "Linear", only_viz=True)
```

![](https://docs.scipy.org/doc/scipy/reference/_images/mgc_plot1_01_00.png)

现在看看检验统计量，p值和MGC图。图中最佳比例用红色“X”表示：

![](https://docs.scipy.org/doc/scipy/reference/_images/mgc_plot2.png)

从这里可以清楚地看到，MGC能够确定输入数据矩阵之间的关系，因为p值非常低，而MGC检验统计信息相对较高。 MGC图显示强线性关系。直观上，这是因为拥有更多的相邻点将有助于识别$x$和$y$之间的线性关系。在这种情况下，最佳比例等于全局比例，在图上用红色点标记。

对于非线性数据集也可以这样做。以下$x$和$y$数列是从非线性模拟得出的：

```python
>>> np.random.seed(12345678)
>>> unif = np.array(np.random.uniform(0, 5, size=100))
>>> x = unif * np.cos(np.pi * unif)
>>> y = unif * np.sin(np.pi * unif) + 0.4 * np.random.random(x.size)
```

模拟的关系图如下：

```python
>>> mgc_plot(x, y, "Spiral", only_viz=True)
```

![](https://docs.scipy.org/doc/scipy/reference/_images/mgc_plot3_01_00.png)

现在，我们可以在下面看到可视化的检验统计量，p值和MGC图。最佳比例在地图上显示为红色“ x”：

```python
>>> stat, pvalue, mgc_dict = multiscale_graphcorr(x, y)
>>> print("MGC test statistic: ", round(stat, 1))
MGC test statistic:  0.2
>>> print("P-value: ", round(pvalue, 1))
P-value:  0.0
>>> mgc_plot(x, y, "Spiral", mgc_dict, only_mgc=True)
```

![](https://docs.scipy.org/doc/scipy/reference/_images/mgc_plot4.png)

从这里可以明显看出，MGC能够再次确定关系，因为p值非常低，并且MGC检验统计量相对较高。 MGC图表示强烈的非线性关系。在这种情况下，最佳比例等于局部比例，在地图上用红色点标记。

# Statistical functions （scipy.stats）

本模块包含大量概率分布和持续增加的统计函数。

每个单变量分布就是rv_continuous(离散变量是rv_discrete)子类中的一个实例：

rv_continuous([])	一般连续随机变量类，用于子类化。

rv_discrete([])	一般离散随机变量类，用于子类化。

rv_histogram([])	生成直方图给出的分布。

## Continuous distributions

| 函数名称 | 概要               | 描述 |
| :------- | ------------------ | ---- |
| alpha    | alpha连续随机变量  |      |
| anglit   | anglit连续随机变量 |      |
|          |                    |      |







## Correlation functions

- f_oneway

  - 单因素方差分析(ANOVA)

  - 使ANOVA检验的p值有效必须满足的前提：

    - 样本独立
    - 每个样本来源于正态分布总体
    - 组内的总体标准偏差相等，即同方差性

    如果这些假设对于给定的数据集不成立，那么仍然可以使用Kruskal-Wallis H检验，但会减少一些效能。

- pearsonr

  - 皮尔逊相关系数与不相关性的p值

  - 皮尔逊相关系数测量两个数据集之间的线性关系。 p值的计算依赖于每个数据集均呈正态分布的假设。像其他相关系数一样，该系数在-1和+1之间变化，其中0表示无相关。 -1或+1的相关性暗示精确的线性关系。正相关表明，随着x的增加，y也随之增加。负相关性表示随着x增加，y减小。

    p值大致表示从不相关的系统中生成的数据集中得到的Pearson相关性与从这些数据集计算得出的一样极端的概率。

  - 相关系数计算方式：
    $$
    r=\frac{\sum{(x-m_x)(y-m_y)}}{\sqrt{\sum(x-m_x)^2\sum(y-m_y)^2}}
    $$
    $m_x$和$m_y$分别是向量$x$和$y$的平均值。

    假设x和y是独立的正态分布（总体相关性是0），样本相关系数的概率密度函数是：
    $$
    f(r)=\frac{(1-r^2)^{(n/2-2)}}{B(1/2, n/2-1)}
    $$
    $n$是样本数量，$B$是beta函数，有时也称为r的精确分布。pearsonr中使用该分布计算p值。分布是取值[-1, 1]的beta分布，形状参数$a=b=n/2-1$。基于Scipy对beta分布的方式，r分布为：

    ```dist = scipy.stats.beta(n/2-1, n/2-1, loc=-1, scale=2```

    pearsonr返回双尾p值。对于给定的具有相关性为r的样本，p值是来自于零相关总体的随机样本x'和y'的相关性abs(r')大于或者等于abs(r)的概率。使用上述`dist`，给定r和长度n的p值为：

    ```p = 2*dist.cdf(-abs(r))```

    当n为2时，上述连续分布的定义不明确。当形状参数a和b接近a = b = 0时，可以将beta分布的极限解释为在r=1和r=-1时具有相等概率质量的离散分布。更直接地，可以观察到，给定数据x=[x1，x2]和y=[y1，y2]，并假设x1!= x2和y1!= y2，r的唯一可能值为1和-1 。因为长度为2的任何样本x'和y'的abs(r')将为1，因此长度为2的样本的两侧p值始终为1。

- spearmanr

  - 计算具有关联p值的Spearman相关系数

  - Spearman等级相关系数是两个数据集之间关系的单调性的非参度量。与Pearson相关性不同，Spearman相关性不假定两个数据集均呈正态分布。像其他相关系数一样，该系数在-1和+1之间变化，0表示无相关。 -1或+1的相关关系意味着确切的单调关系。正相关表明，随着x的增加，y也随之增加。负相关性表示随着x增加，y减小。

    p值大致表示不相关系统的数据集的Spearman相关性与从这些数据集计算得出的Spearman相关性至少一样极端的可能性。 样本量大于500左右时p值比较可靠。

- pointbiserialr

  - 点二列相关系数和p值。

  - 点二列相关用于测量二进制变量x和连续变量y之间的关系。像其他相关系数一样，该系数在-1和+1之间变化，其中0表示无相关。 -1或+1的相关性意味着确定的关系。

  - `n-1`自由度的t检验，等价于pearsonr。

    点二列相关值由公式计算：
    $$
    r_{pb}=\frac{\bar{Y_1}-\bar{Y_0}}{s_y}\sqrt{\frac{N_1N_2}{N(N-1)}}
    $$
    $Y_0$和$Y_1$分别是编码0和1的度量观测值的平均值；$N_0$和$N_1$分别是编码0和1的观测数量；$N$是观测总数，$s_y$是所有度量观测值的标准差。

    $r_{pb}$与0的显著差异完全等价于两组间的显著差异。因此$N-2$自由度水平的独立组间t检验用来检验$r_{pb}$是否为0。两个独立组间比较t统计值与$r_{pb}$的关系由下式给出：
    $$
    t=\sqrt{N-2}\frac{r_{pb}}{\sqrt{1-r_{pb}^2}}
    $$

- Kendalltau

  - 计算Kendall的tau值，等级相关系数的度量。

  - Kendall的tau值度量两个排序间的相似性。如果两个序列完全一致，则Kendall's tau值为1，两个毫不相关的序列的Kendall's tau值为0，而两个互逆的序列的Kendall's tau系数为-1。1945的"tau-b"版本考虑了秩，1938的“tab-a”版没有考虑秩。

  - Kendall tau定义为：
    $$
    tau = \frac{P-Q}{\sqrt{(P+Q+T)(P+Q+U)}}
    $$
    $P$是相同对的数量，$Q$是不同对的数量，$T$是只在x中秩数量，$U$是只在y中秩数量，如果秩发生在x和y同一对中，都不计入。

- weightedtau

  - 计算Kendall的加权$\tau$。

  - 加权$\tau$中高权重的交换比低权重的交换更重要。默认参数计算索引$\tau_h$的加法双曲形式，这为重要的不重要的元素间保持了平衡。

    权重通过秩数组定义，为每个元素赋予非负秩，基于元素的秩评估权重。一个交换的权重即为交换元素秩的权重的和或积。默认参数计算$\tau_h$：具有$r$和$s$（从0开始）的秩的元素的交换的权重$1/(r+1)+1/(s+1)$。

    只有重要性标准存在情况下，指定秩数组才有意义。如果没有特定的排序，加权$\tau$就是通过(x, y)和(y, x)字母降序排序的平均值。这是默认参数的计算方式。

    注意，如果计算秩数组的加权$\tau$，而不是得分（即，很大的值表示代的排序）必须对秩取负值，这样高秩的元素才会得到高值。

  - 

  



































