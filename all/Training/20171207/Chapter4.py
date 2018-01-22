# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 11:27:49 2017

@author: Admin
"""
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# サンプルデータ作成
df = pd.DataFrame([
        ['green','M','10.1','class1'],
        ['red','L','13.5','class2'],
        ['blue','XL','15.3','class1'],
        ])
# 列名を指定
df.columns = ['color','size','price','classlabel']

# Tシャツのサイズを整数に変換
size_mapping = {'XL': 3,'L': 2,'M': 1}
df['size'] = df['size'].map(size_mapping)

# クラスラベルを整数に変換
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)

# Tシャツの色、サイズ、価格を抽出
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

# one-hot エンコーダの作成
ohe = OneHotEncoder(categorical_features=[0])
# one-hot エンコーディングを実行
ohe.fit_transform(X).toarray()
X