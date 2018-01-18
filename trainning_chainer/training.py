# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 13:05:56 2017

@author: Matsuura
"""

import numpy as np
import pandas as pd
from chainer import datasets, iterators, optimizers, training
from chainer import Chain
from chainer.training import extensions
from sklearn.preprocessing import MinMaxScaler
import chainer.functions as F
import chainer.links as L



class Model(Chain):
    
    # 初期化
    def __init__ (self):
        super(Model, self).__init__(
            # 各層のニューロン数を定義
            # 本案件では「特徴量 < ニューロン数」となる方針
            l1 = L.Linear(10, 40),
            l2 = L.Linear(40, 40),
            l3 = L.Linear(40, 40),
            l4 = L.Linear(40, 40),
            l5 = L.Linear(40, 2)
            )
    
    
    # プーリング　（Relu関数）
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        return self.l5(h4)
    

def conv(data):
    # 抽出値を初期化
    X = []
    
    Y = []
    
    # numpy変換用
    data_array = data.as_matrix()
    
    # 説明変数（20）と目的変数（1）を分割する。
    for j in data_array:
        x_split = np.hsplit(j, [10, 11])
        X.append(x_split[0].astype(np.float32))
        Y.append(x_split[1].astype(np.int32))
    
    X = np.array(X)
    Y = np.ndarray.flatten(np.array(Y))
    
    # テストデータのスケーリング
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    # 訓練データ、検証データを無作為に8：2で分割する。
    train, test = datasets.split_dataset_random(datasets.TupleDataset(X, Y), 32000)
    
    # 訓練データ、検証データを返却する
    return train, test


def main():

    # ハイパーパラメータを定義
    # 学習回数
    epoch = 30
    # バッチサイズ
    batchsize = 100
    # 学習率
    learning_rate = 0.01
    
    # 入力ファイルを読み込む (pandasで読み込んだ方が早いため、pandasで読込後にnumpyへ変換する)
    data = pd.read_csv('C:\wk\UCI_add1/bank-full.csv')
     # chainer読込用のデータ型に変換する
    train, test = conv(data)
    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize,repeat=False, shuffle=False)
    
    # モデル定義
    model = L.Classifier(Model())
    # 最適化
    optimizer = optimizers.SGD(learning_rate)     # 確率的勾配降下
    optimizer.setup(model)
    
    # trainerを使用し深層学習を行う
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'))

    # 学習後のモデルに検証用データを投入し検証を行う
    trainer.extend(extensions.Evaluator(test_iter, model))
    
    # 学習結果を視覚化する。
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], file_name='loss.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 
                                           'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    
    trainer.run()

if __name__ == '__main__': 
    main()