# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:27:27 2017

@author: Matsuura
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# グラフ出力
def plot_feature_importances(model):
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_)
  #  plt.yticks(np.arange(n_features),label)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
# 項目名
#label = np.array(['age','job','education','day_of_week','duration','campaign','pdays','cons.conf.idx','euribor3m','nr.employed'])

# 文字列は既に置換済
# unknownは最大件数の値に置換。
# 小数点以下の値は切捨て
df_bank = pd.read_csv('C:\wk\UCI_add1/bank-full.csv')

# 特徴量とクラスラベルを別々に抽出
X,y = df_bank.iloc[:,0:-1].values, df_bank.iloc[:,-1].values
 
# テストデータセットを訓練セットとテストセットに分割 (7:3)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

# グリッドサーチを用いて、ランダムフォレストの特徴量、最適パラメータを抽出
parameters = {
        'n_estimators':[175, 180, 185],
        'max_depth':[20, 21, 22]
        }

grid = GridSearchCV(RandomForestClassifier(class_weight='balanced'), parameters)
grid.fit(X_train,y_train)

print ("Training Score: {:.2f}".format(grid.score(X_train, y_train)))   # 訓練セット正解率
print ("Test Score: {:.2f}".format(grid.score(X_test, y_test)))         # テストセット正解率
print ("Best Parameters: {}".format(grid.best_params_))

# 結果をグラフ出力
#plot_feature_importances(grid)

# モデル評価
score = cross_val_score(grid.best_estimator_, X_test, y_test, cv=5)
print score.mean()

print "------------Confusion Matrix--------------"
y = grid.predict(X_train)
print confusion_matrix(y_train, y) 