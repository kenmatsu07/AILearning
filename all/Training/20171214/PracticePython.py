# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 09:13:30 2017

@author: Admin
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn

# 文字列は既に置換済 (ランダムフォレストは値差の考慮が不要のため)
# unknownは一律「99」で変換。
df_bank = pd.read_csv('C:/wk/ucl/bank-additional_forPython.csv')

# 特徴量とクラスラベルを別々に抽出
X,y = df_bank.iloc[:,0:-1].values, df_bank.iloc[:,-1].values
 
# テストデータセットを訓練セットとテストセットに分割 (8:2)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# グリッドサーチを利用し学習
# 精度向上させるため、パイプラインを使用し訓練セットを訓練用、検証用に分割する。
pipe = Pipeline([("scaler", MinMaxScaler()),("svm", SVC(class_weight='balanced'))])
pipe.fit(X_train,y_train)
parameters = {'svm__C':[0.001, 0.01, 0.1, 1, 10, 100],'svm__gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, parameters, cv=5)
grid.fit(X_train,y_train)
print ("Best Accuracy: {:.2f}".format(grid.best_score_))
print ("Best Parameters: {}".format(grid.best_params_))

# モデル評価
score = cross_val_score(grid.best_estimator_, X, y, cv=5)
print score.mean()

print "------------Confusion Matrix--------------"
y = grid.predict(X_train)
print confusion_matrix(y_train, y) 

