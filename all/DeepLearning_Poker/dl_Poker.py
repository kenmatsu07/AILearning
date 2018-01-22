# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:46:57 2018

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
            l1 = L.Linear(10, 20),
            l2 = L.Linear(20, 20),
            l3 = L.Linear(20, 10)
            )
    
    
    # プーリング　（Relu関数）
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
    

def main():

    # ハイパーパラメータを定義
    # 学習回数
    epoch = 100
    # バッチサイズ
    batchsize = 500
    # 学習率
    learning_rate = 0.01
    
    # 入力ファイルを読み込む
    train = np.loadtxt('C:/wk/UCI_add2/data/poker-hand-training-true.csv')
    test = np.loadtxt('C:/wk/UCI_add2/data/poker-hand-testing.csv')
    
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