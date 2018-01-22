# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:51:10 2017

@author: Matsuura
""銀行データ解析課題＿改良版"""
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn

# テスト用ファイルを読み込む
df_bank = pd.read_csv('C:/wk/ucl/bank-additional-work_20171212.csv')

# ワンホットエコーディングを実行
data_dummies = pd.get_dummies(df_bank)

# 説明変数、目的変数を定義
X = data_dummies.iloc[:,0:-1].values
y = data_dummies.iloc[:,-1].values
  
# テストデータセットを訓練セットとテストセットに分割 (7:3)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
 
# グリッドサーチを利用し学習
# 精度向上させるため、パイプラインを使用し訓練セットを訓練用、検証用に分割する。
pipe = make_pipeline(StandardScaler(),LogisticRegression())
parameters = {'logisticregression__C':[0.01, 0.01, 0.1, 1]}
grid = GridSearchCV(pipe, parameters, cv=5)
grid.fit(X_train,y_train)

# 特徴量、訓練データ正解率、テストデータ正解率、最適パラメータをそれぞれ出力
print ("X_train.shape: {}".format(X_train.shape))
print ("Training Score: {:.2f}".format(grid.score(X_train, y_train)))
print ("Test Score: {:.2f}".format(grid.score(X_test, y_test)))
print ("Best_estimator:\n{}".format(grid.best_estimator_))
print ("Best Parameters: {}".format(grid.best_params_))

# グラフ出力
plt.plot(X_train.min(axis=0),'o',label="min")
plt.plot(X_train.max(axis=0),'^',label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")