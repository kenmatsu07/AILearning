# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:52:19 2018

@author: Admin
"練習：ポーカー　（回帰問題）"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import itertools

# データを読み込み
df_train = pd.read_csv('C:/wk/UCI_add2/data/poker-hand-training-true.csv')
df_test = pd.read_csv('C:/wk/UCI_add2/data/poker-hand-testing.csv')

# ループ分で使用する変数を定義
# 訓練データのレコード数
train_length = len(df_train)

# 目的変数はマークによる役の強さまでは求めていないため、1行内のカードマークを入れ替えることで学習を深める
for i,v in df_train.iterrows():
        
    # カードマークを表す項目値を抽出
    cardmark = v['S1', 'S2', 'S3', 'S4', 'S5']
    # カードマークの順列を作成
    s_cardmark = list(itertools.permutations(cardmark))
    X = pd.DataFrame([], colums=[], index = train_length + i)
    df_train.append()

# 特徴量とクラスラベルを別々に抽出
X_train, y_train = df_train.iloc[:,0:-1].values, df_train.iloc[:,-1].values
X_test, y_test = df_test.iloc[:,0:-1].values, df_test.iloc[:,-1].values    

# グリッドサーチを用いて、ランダムフォレストの特徴量、最適パラメータを抽出
forest = RandomForestClassifier(n_estimators = 220, max_depth=14)
forest.fit(X_train, y_train)

#grid = GridSearchCV(forest, parameters)
#grid.fit(X_train,y_train)

print ("Training Score Grid: {:.2f}".format(forest.score(X_train, y_train)))  # 訓練セット正解率
print ("Test Score Grid: {:.2f}".format(forest.score(X_test, y_test)))        # テストセット正解率
#print ("Best Accuracy: {:.2f}".format(forest.best_score_))
#print ("Best Parameters: {}".format(forest.best_params_))

# モデル評価
#score = cross_val_score(forest.best_estimator_, X_test, y_test, cv=5)
#print score.mean()
