# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 14:13:29 2017

@author: Admin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

# =============================================================================
# データセットの生成
X,y = mglearn.datasets.make_forge()
# # データセットをプロット
# mglearn.discrete_scatter(X[:, 0],X[:, 1],y)
# plt.legend(["Class 0"],"Class 1",loc=4)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# print ("X.shape: {}".format(X.shape))
# =============================================================================

# =============================================================================
# X,y = mglearn.datasets.make_wave(n_samples=40)
# plt.plot(X,y,'o')
# plt.ylim(-3,3)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# =============================================================================

# =============================================================================
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(X_train,y_train)
# 
# # テストデータの予測
# print ("Test set predictions: {}".format(clf.predict(X_test)))
# # モデルの汎化性能を評価
# print ("Test set accuracy: {:.2f}".format(clf.score(X_test,y_test)))
# =============================================================================

# =============================================================================
# fig,axes = plt.subplots(1,3,figsize=(10,3))
# for n_neighbors,ax in zip([1,3,9],axes):
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
#     mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=.4)
#     mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
#     ax.set_title("{} neighbor(s)".format(n_neighbors))
#     ax.set_xlabel("feature 0")
#     ax.set_ylabel("feature 1")
# axes[0].legend(loc=3)
# =============================================================================


# =============================================================================
# cancer = load_breast_cancer()
# X_train,X_test,y_train,y_test = train_test_split(
#         cancer.data,cancer.target,stratify=cancer.target,random_state=66)
# 
# training_accuracy = []
# test_accuracy = []
# # n_neighbors を1から10まで試す
# neighbors_settings = range(1,11)
# 
# for n_neighbors in neighbors_settings:
#     # モデルを構築
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#     clf.fit(X_train,y_train)
#     # 訓練セットモデルを記録
#     training_accuracy.append(clf.score(X_train,y_train))
#     # 汎化精度を記録
#     test_accuracy.append(clf.score(X_test,y_test))
#     
# plt.plot(neighbors_settings,training_accuracy,label="training accuracy")
# plt.plot(neighbors_settings,test_accuracy,label="test accuracy")
# plt.ylabel("Accuracy")
# plt.ylabel("n_neighbors")
# plt.legend()
# =============================================================================


# =============================================================================
# X,y = mglearn.datasets.make_wave(n_samples=40)
# waveデータセットを訓練セットとテストセットに分割
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
# 
# 3つの最近傍点を考慮するように設定しモデルのインスタンスを生成
# reg = KNeighborsRegressor(n_neighbors=3)
# 訓練データと訓練ターゲットを用いてモデルを学習させる
# reg.fit(X_train,y_train)
# 
# print ("Test set precictions:\n{}".format(reg.predict(X_test)))
# =============================================================================


fig,axes = plt.subplots(1,3,figsize=(15,4))
# -3から3までの間に1000点のデータポイントを作る
line = np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors,ax in zip([1,3,9],axes):
     # 1,3,9 近傍点で予測
     reg = KNeighborsRegressor(n_neighbors=n_neighbors)
     reg.fit(X_train,y_train)
     ax.plot(line,reg.predict(line))
     ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
     ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize=8)
     
     ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
             n_neighbors,reg.score(X_train,y_train),
             reg.score(X_test,y_test)))
     ax.set_xlabel("Feature")
     ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"],loc="best")