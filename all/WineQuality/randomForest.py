# -*- coding: utf-8 -*-１
"""
Created on Wed Jan 17 10:37:45 2018

@author:Matsuura
"練習：ワイン品質　（回帰問題）"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import mglearn
import seaborn as sns
from sklearn.cluster import DBSCAN

## 特徴量（上位5）を抽出
#def selfeatures(X_train):
#    
#    # 相関係数を抽出
#    plt.figure(figsize=(15,15))
#    seaborn.heatmap(df_wine.corr(), annot=True)
#    
#    # 特徴量を絞り込む
#    select = SelectFromModel(
#            RandomForestClassifier(n_estimators = 1000),
#            threshold="median")
#    select.fit(X_train, y_train)
#    # 訓練セットを変換
#    X_train_selected = select.transform(X_train)

# ヒストグラムを生成
def makefeaturehistpair():
    
    # 外れ値
    df_wine.plot(kind='hist', bins=50, subplots=True)
    
    # 散布図を作成
    sns.pairplot(df_wine, hue = "quality", diag_kind="kde")
    (df_wine.sort_values('fixed acidity').
     plot.barh(subplots=True, layout=(3, 4), sharex=False, legend=False))
    plt.show()
    

# 箱ひげ図を生成
def makeboxplot():
    fig, axes = plt.subplots(figsize=(10,10))
    
    for i, (n, g) in enumerate(df_wine.groupby('quality')):
        sns.boxplot(data=g.iloc[:, 0:-1], ax=axes[i])
        axes[i].set_ylabel(n)
    
#    ax = sns.boxplot(111)
#    ax.boxplot(wine_dataframe)

 
# 外れ値を検出
def getnoise():

    # 外れ値検出の精度向上のため、テストデータのスケーリングを行う
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    # 外れ値を視覚
    for eps in [0.2, 0.4, 0.6, 0.8]:
        
        print ("\neps={}".format(eps))
        dbscan = DBSCAN(min_samples=3, eps=eps)
        labels = dbscan.fit_predict(X_scaled)
        print ("Clusters present: {}".format(np.unique(labels)))
        print ("Clusters sizes: {}".format(np.bincount(labels + 1)))
    
# マトリックスを生成
#def getfeturematrix():
#    
#    # マトリックスを生成
#    pd.scatter_matrix(wine_dataframe, c=y_train, figsize = (30, 30) ,marker = '.',
#                            hist_kwds={'bins': 40}, s=100, alpha=.8 ,cmap=mglearn.cm3)    

# 特徴量の重要度を視覚化
#def getcorrelation():
#    # -- ランダムフォレスト -- (パラメータは任意の値)
#    forest = RandomForestClassifier(n_estimators = 1000, max_depth=30, class_weight='balanced')
#    forest.fit(X_train, y_train)
#    
#    # グラフを出力
#    n_features = X_train.shape[1]
#    plt.barh(range(n_features), forest.feature_importances_, align='center')
#    plt.yticks(np.arange(n_features),feature_names)
#    plt.xlabel("Feature importance")
#    plt.ylabel("Feature")
        

# データを読み込み
df_wine = pd.read_csv('C:\wk\UCI_WineQuality\data/winequalityForLearning.csv')

# 要約統計量を抽出
print (df_wine.iloc[:,0:-1].describe())

# 説明変数と目的変数を抽出
X,y = df_wine.iloc[:,0:-1].values, df_wine.iloc[:,-1].values

# テストデータセットを訓練セットとテストセットに分割 (8:2)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# 説明変数の項目名
feature_names = np.array(['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
                          'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'])


wine_dataframe = pd.DataFrame(X_train, columns = feature_names)
    
# 外れ値検出関数の呼び出し
getnoise()

# ヒストグラム生成関数の呼び出し
makefeaturehistpair()

# 箱ひげ図生成関数の呼び出し
makeboxplot()

# マトリックス生成関数の呼び出し
#getfeturematrix()

# 相関係数を抽出
#getcorrelation()

## グリッドサーチを用いて、ランダムフォレストの特徴量、最適パラメータを抽出
#parameters = {
#        'n_estimators':[1000, 1020, 1040],
#        'max_depth':[35]
#        }
#
#grid = GridSearchCV(RandomForestClassifier(), parameters)
#grid.fit(X_train,y_train)
#
#print ("Training Score: {:.2f}".format(grid.score(X_train, y_train)))   # 訓練セット正解率
#print ("Test Score: {:.2f}".format(grid.score(X_test, y_test)))         # テストセット正解率
#print ("Best Parameters: {}".format(grid.best_params_))                 # 最適パラメータ
#
## モデル評価
#score = cross_val_score(grid.best_estimator_, X_test, y_test, cv=5)
#print score.mean()
#
#print "------------Confusion Matrix--------------"
#y = grid.predict(X_train)
#print confusion_matrix(y_train, y) 