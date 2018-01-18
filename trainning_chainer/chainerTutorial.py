# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 08:15:17 2017

@author: Matuura
"""

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.datasets import mnist
import matplotlib.pyplot as plt

# データセットをダウンロード
train, test = mnist.get_mnist(withlabel=True, ndim=1)

# データの例示
X, y = train[0]
plt.imshow(X.reshape(28, 28), cmap='gray')
plt.show()
print('label:', y)

