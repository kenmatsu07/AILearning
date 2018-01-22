# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 09:13:30 2017

@author: Matsuura
""銀行データ解析課題＿改良版"""
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn

# =============================================================================
# # モデル出力用グラフ (グリッドサーチ)
# def plot_feature_importances_grid(model):
#  plt.plot(X_train.min(axis=0),'o',label="min")
#  plt.plot(X_train.max(axis=0),'^',label="max")
#  plt.legend(loc=4)
#  plt.xlabel("Feature index")
#  plt.ylabel("Feature magnitude")
#  plt.yscale("log")
# 
# =============================================================================
# モデル出力用グラフ (ランダムフォレスト)
def plot_feature_importances_forest(model):
     n_features = X_train.shape[1]
     plt.barh(range(n_features), model.feature_importances_, align='center')
     plt.xlabel("Feature importance")
     plt.ylabel("Feature")

# テスト用ファイルを読み込む
df_bank = pd.read_csv('C:/wk/ucl/bank-additional-work_20171212.csv')

# ワンホットエコーディングを実行
data_dummies = pd.get_dummies(df_bank)

# 説明変数、目的変数を定義
X = data_dummies.iloc[:,0:-1].values
y = data_dummies.iloc[:,-1:].values

# テストデータセットを訓練セットとテストセットに分割 (7:3)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
 
# -- グリッドサーチ --
pipe = Pipeline([("scaler", MinMaxScaler()),("svm", SVC())])
pipe.fit(X_train,y_train)
# Memo：gammaが小さいほど決定境界が滑らかになり、逆に大きいほど複雑になっていく。
parameters = {'svm__C':[0.001, 0.01, 0.01, 0.1, 1],'svm__gamma':[0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(pipe, parameters, cv=5, scoring = 'accuracy')
grid.fit(X_train,y_train)

print ("X_train.shape: {}".format(X_train.shape))                           # 特徴量
print ("Training Score Grid: {:.2f}".format(grid.score(X_train, y_train)))  # 訓練セット正解率
print ("Test Score Grid: {:.2f}".format(grid.score(X_test, y_test)))        # テストセット正解率
print ("Best estimator Grid:\n{}".format(grid.best_estimator_))             # 最適モデル
print ("Best Parameters Grid:{}".format(grid.best_params_))                 # 最適パラメータ

# グラフを出力
#plot_feature_importances_grid(grid)

# -- ランダムフォレスト --
forest = RandomForestClassifier(n_estimators = 100, max_depth=4)
forest.fit(X_train, y_train)

print ("Training Score Forest: {:.2f}".format(grid.score(X_train, y_train)))    # 訓練セット正解率
print ("Test Score Fors: {:.2f}".format(grid.score(X_test, y_test)))            # テストセット正解率

# グラフを出力
plot_feature_importances_forest(forest)
