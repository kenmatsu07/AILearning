# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:38:06 2017

@author: Admin
"""
# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

# 文字列は既に置換済
# unknownは最大件数のグループに置換。
# 削除項目は「default」、「duration」、「campaign」、「pdays」、「cons.price.idx」
# 小数点以下の値は切捨て
df_bank = pd.read_csv('C:\wk\UCI_add1/bank-full_01.csv')

# 特徴量とクラスラベルを別々に抽出
X,y = df_bank.iloc[:,0:-1].values, df_bank.iloc[:,-1].values
 
# テストデータセットを訓練セットとテストセットに分割 (8:2)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# 相関係数を抽出
plt.figure(figsize=(15,15))
seaborn.heatmap(df_bank.corr(), annot=True)

# 特徴量を絞り込む
select = SelectFromModel(
        RandomForestClassifier(n_estimators = 1000),
        threshold="median")
select.fit(X_train, y_train)
# 訓練セットを変換
X_train_selected = select.transform(X_train)

# 使用されている特徴量を抽出する。
mask = select.get_support()
print(mask)
# マスクを可視化する。
plt.matshow(mask.reshape(1, -1), cmap = 'gray_r')
plt.xlabel("Test")