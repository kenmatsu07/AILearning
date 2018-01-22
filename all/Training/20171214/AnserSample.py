
"""
Created on Wed Dec 13 14:58:47 2017
@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import grid_search
from sklearn.model_selection import train_test_split

# XORゲート形式のデータセットを作成
np.random.seed(0)
# 標準正規分布に従う乱数で200行2列の行列を生成
X_xor = np.random.randn(200, 2)

# 二つの引数に対して排他的論理和を実行
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
# 真の場合は1、偽の場合は-1を割り当てる
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')

# 作成したデータセットをグラフ出力
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

# 軸範囲を設定
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 学習データとテストデータの分割
X_train, X_test, y_train, y_test = train_test_split(X_xor, y_xor, random_state=42)

SVM_parameters = {'C': [0.1, 1, 20, 50, 80], 'gamma': [0.0001, 0.001, 0.01, 0.1]}

clf = grid_search.GridSearchCV(svm.SVC(), SVM_parameters,  scoring='accuracy')                      
# clf3 = grid_search.GridSearchCV(svm.SVC(), SVM_parameters,  scoring='precision')
clf.fit(X_train, y_train)
print clf.best_params_

# ベストパラメータのモデル
best = clf.best_estimator_
print best

print "------------Confusion Matrix--------------"
y = best.predict(X_train)
print confusion_matrix(y_train, y) 

x1_min, x1_max = X_xor[:, 0].min() - 1,X_xor[:, 0].max() + 1    # Xの1次元目の最小と最大を取得
x2_min, x2_max = X_xor[:, 1].min() - 1, X_xor[:, 1].max() + 1   # Xの2次元目の最小と最大を取得

# x1_min から x1_max まで、x2_min から x2_max までの h 刻みの等間隔な格子状配列を生成
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max,0.1), np.arange(x2_min, x2_max,0.1))

# .ravel()で一次元配列に変換
num = len(xx1.ravel())

# メッシュ状の各点に対して予測
#np.c_はxx1とxx2とnp.ones()を合体させる
Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape) # 配列形式変更

# グラフ作成
cmap_light = ListedColormap(['#FFAAAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FFAAAA','#AAAAFF'])
plt.pcolormesh(xx1, xx2, Z, cmap=cmap_light)                # 学習結果をプロット
print "-------------------------------"
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
plt.show()