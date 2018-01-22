# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:14:07 2017

@author: Matsuura
""サンプル＿決定木ver"""

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
import os

# --決定木の出力テスト--
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#         cancer.data, cancer.target, stratify = cancer.target, random_state = 42)
# 
# # 決定木を作成
# tree = DecisionTreeClassifier(max_depth = 4, random_state = 0)
# tree.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
# 
# # 特徴量の重要度を出力
# print ("Feature importances:\n{}".format(tree.feature_importances_))
# def plot_feature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(range(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), cancer.feature_names)
#     plt.xlabel("Feature importance")
#     plt.ylabel("Feature")
# plot_feature_importances_cancer(tree)
#
# # 決定木を可視化する。
# export_graphviz(tree, out_file = "tree.dot", class_names=["malignant", "benign"],
#                 feature_names = cancer.feature_names, impurity = False, filled = True)
# with open("tree.dot") as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)
# --ここまで--

# 決定木は外挿不可であることの確認
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH,
                                      "ram_price.csv"))
# ワーク用
# plt.semilogy(ram_prices.date, ram_prices.price)
# plt.xlabel("Year")
# plt.ylabel("Proce in $/Mbyte")

# 過去データを用いて2000年以降の価格を予想する。
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# 日付に基づいて価格予測
X_train = data_train.date[:, np.newaxis]
# データとターゲットの関係を単純にするために対数変換
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train,y_train)
linear_reg = LinearRegression().fit(X_train,y_train)

# 全ての価格を予想
X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# 対数変換をキャンセルするために逆変換
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

# ram_prices以下をグラフ出力
plt.semilogy(data_train.date, data_train.price, label = "Training data")
plt.semilogy(data_test.date, data_test.price, label = "Test data")
plt.semilogy(ram_prices.date, price_tree, label = "Tree Prediction")
plt.semilogy(ram_prices.date, price_lr, label = "Liniar Prediction")
plt.legend()