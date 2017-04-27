"""
===========================================================
Plot Ridge coefficients as a function of the regularization
===========================================================
显示共线性对估计器系数的影响。
：class：`Ridge`回归是这个例子中使用的估计量。
每个颜色代表系数向量的不同特征，并且被显示为正则化参数的函数。

某些矩阵，目标变量的轻微改变，能导致权重的巨大改变。
这种情况下，设置一定的正则化（alpha）是有用的，能减少这种变化（噪声）

当alpha很大时，正则化效应导致平方损失函数和系数趋于0.
当alpha趋于0时，系数会有大的震荡。
在实践中，有必要调整alpha，在这两者之间保持平衡。
"""


# Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

###############################################################################
# Compute paths

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)

###############################################################################
# Display results

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()