# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:07:41 2017

@author: Matsuura
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

# モデル出力用グラフ (グリッドサーチ)
def plot_feature_importances_grid(model):
  plt.plot(X_train.min(axis=0),'o',label="min")
  plt.plot(X_train.max(axis=0),'^',label="max")
  plt.legend(loc=4)
  plt.xlabel("Feature index")
  plt.ylabel("Feature magnitude")
  plt.yscale("log")

# 文字列は既に置換済
# unknownは最大件数の値に置換。
# 小数点以下の値は切捨て
df_bank = pd.read_csv('C:/wk/ucl_addPractice/bank-full.csv')

# ワンホットエンコーディングするデータ項目
# label = np.array(['duration','nr.employed','euribor3m', 'cons.conf.idx', 'pdays'])

# ワンホットエンコーディングを実行
data_dummies = pd.get_dummies(df_bank)

# 項目並び順をランダムにする　
random = np.random.permutation(data_dummies)

# 特徴量とクラスラベルを別々に抽出
X,y = data_dummies.iloc[:,0:-1].values, data_dummies.iloc[:,-1].values

# テストデータセットを訓練セットとテストセットに分割 (7:3)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

# -- グリッドサーチ --
pipe = Pipeline([("scaler", MinMaxScaler()),
                 ("svm", SVC(class_weight='balanced'))])
pipe.fit(X_train,y_train)
parameters = {'svm__C':[70, 75],'svm__gamma':[110, 120]}
grid = GridSearchCV(pipe, parameters)
grid.fit(X_train,y_train)
print ("Training Score Grid: {:.2f}".format(grid.score(X_train, y_train)))  # 訓練セット正解率
print ("Test Score Grid: {:.2f}".format(grid.score(X_test, y_test)))        # テストセット正解率
print ("Best Accuracy: {:.2f}".format(grid.best_score_))
print ("Best Parameters: {}".format(grid.best_params_))

# グラフを出力
plot_feature_importances_grid(grid)

# モデル評価
score = cross_val_score(grid.best_estimator_, X_test, y_test, cv=5)
print score.mean()

print "------------Confusion Matrix--------------"
y = grid.predict(X_train)
print confusion_matrix(y_train, y) 
