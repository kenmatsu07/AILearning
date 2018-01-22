# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:32:27 2017

@author: Matsuura
"サンプル＿ディープラーニング"""

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn

# テストデータをセット (隠れ層：10)
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=42)

mlp = MLPClassifier(solver = 'lbfgs', random_state=0, hidden_layer_sizes = [10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

# グラフ出力
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# テストデータをセット (隠れ層：10、非線形活性化関数に「tanh」を使用)
mlp = MLPClassifier(solver = 'lbfgs', activation = 'tanh',random_state=0, hidden_layer_sizes = [10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

# グラフ出力
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")