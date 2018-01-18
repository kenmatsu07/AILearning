# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:27:27 2017

@author: Matsuura
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# モデル出力用グラフ (ランダムフォレスト)
def plot_feature_importances_forest(model):
     n_features = X_train.shape[1]
     plt.barh(range(n_features), model.feature_importances_, align='center')
     plt.yticks(np.arange(n_features),label)
     plt.xlabel("Feature importance")
     plt.ylabel("Feature")
     
# 項目名
label = np.array(['age','job','education','day_of_week','duration','campaign','pdays','cons.conf.idx','euribor3m','nr.employed'])

# 文字列は既に置換済
# unknownは最大件数の値に置換。
# 小数点以下の値は切捨て
df_bank = pd.read_csv('C:\wk\UCI_add1/bank-full.csv')

# 特徴量とクラスラベルを別々に抽出
X,y = df_bank.iloc[:,0:-1].values, df_bank.iloc[:,-1].values
 
# テストデータセットを訓練セットとテストセットに分割 (7:3)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

# -- ランダムフォレスト --
forest = RandomForestClassifier(n_estimators = 200, max_depth=5, class_weight='balanced')
forest.fit(X_train, y_train)

print ("Training Score Forest: {:.2f}".format(forest.score(X_train, y_train)))    # 訓練セット正解率
print ("Test Score Fors: {:.2f}".format(forest.score(X_test, y_test)))            # テストセット正解率

# グラフを出力
plot_feature_importances_forest(forest)

print "------------Confusion Matrix--------------"
y = forest.predict(X_train)
print confusion_matrix(y_train, y) 