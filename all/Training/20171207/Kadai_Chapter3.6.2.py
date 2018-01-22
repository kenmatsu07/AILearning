# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 09:52:17 2017

@author: Admin
"""
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Irisデータセットをロード
iris = datasets.load_iris()
# 3、4列目の特徴量を抽出
X = iris.data[:, [2,3]]
# クラスラベルを取得
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(
        X,y,size=0.3,random_state=0)

# エントロピーを指標とするランダムフォレストのインスタンスを生成
forest = RandomForestClassifier(criterion='entropy',
                                n_esimators=10, randaom_state=1, n_jobs=2)

X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))

# ランダムフォレストのモデルにトレーニングデータを適合させる
forest.fit(X_train,y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest,
                      test_idx=range(105,150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()