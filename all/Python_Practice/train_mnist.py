# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:58:26 2017

@author: Matsuura
"""
 #!/usr/bin/env python  

 #!/usr/bin/env python 
 

from __future__ import print_function 
import argparse 
import chainer 
import chainer.functions as F 
import chainer.links as L 
from chainer import training 
from chainer.training import extensions
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import sys
from seaborn import heatmap

# Network definition 
class MLP(chainer.Chain): 

    def __init__(self, n_units, n_out): 
        super(MLP, self).__init__() 
        with self.init_scope(): 
            # the size of the inputs to each layer will be inferred 
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units 
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units 
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out 


    def __call__(self, x): 
        h1 = F.relu(self.l1(x)) 
        h2 = F.relu(self.l2(h1)) 
        return self.l3(h2) 
    
def main(): 
    parser = argparse.ArgumentParser(description='Chainer example: MNIST') 
    parser.add_argument('--batchsize', '-b', type=int, default=100, 
                        help='Number of images in each mini-batch') 
    parser.add_argument('--epoch', '-e', type=int, default=20, 
                        help='Number of sweeps over the dataset to train') 
    parser.add_argument('--frequency', '-f', type=int, default=-1, 
                        help='Frequency of taking a snapshot') 
    parser.add_argument('--gpu', '-g', type=int, default=-1, 
                        help='GPU ID (negative value indicates CPU)') 
    parser.add_argument('--out', '-o', default='result', 
                        help='Directory to output the result') 
    parser.add_argument('--resume', '-r', default='', 
                        help='Resume the training from snapshot') 
    parser.add_argument('--unit', '-u', type=int, default=1000, 
                        help='Number of units') 
    parser.add_argument('--noplot', dest='plot', action='store_false', 
                        help='Disable PlotReport extension') 
    args = parser.parse_args() 


    print('GPU: {}'.format(args.gpu)) 
    print('# unit: {}'.format(args.unit)) 
    print('# Minibatch-size: {}'.format(args.batchsize)) 
    print('# epoch: {}'.format(args.epoch)) 
    print('') 


    # Set up a neural network to train 
    # Classifier reports softmax cross entropy loss and accuracy at every 
    # iteration, which will be used by the PrintReport extension below. 
    model = L.Classifier(MLP(args.unit, 10)) 
    if args.gpu >= 0: 
        # Make a specified GPU current 
        chainer.cuda.get_device_from_id(args.gpu).use() 
        model.to_gpu()  # Copy the model to the GPU 


    # Setup an optimizer 
    optimizer = chainer.optimizers.Adam() 
    optimizer.setup(model) 


    # Load the MNIST dataset 
    train, test = chainer.datasets.get_mnist() 


    train_iter = chainer.iterators.SerialIterator(train, args.batchsize) 
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, 
                                                 repeat=False, shuffle=False) 


    # Set up a trainer 
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu) 
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out) 


    # Evaluate the model with the test dataset for each epoch 
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu)) 


    # Dump a computational graph from 'loss' variable at the first iteration 
    # The "main" refers to the target link of the "main" optimizer. 
    trainer.extend(extensions.dump_graph('main/loss')) 


    # Take a snapshot for each specified epoch 
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency) 
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch')) 


    # Write a log of evaluation statistics for each epoch 
    trainer.extend(extensions.LogReport()) 


    # Save two plot images to the result dir 
    if args.plot and extensions.PlotReport.available(): 
        trainer.extend( 
            extensions.PlotReport(['main/loss', 'validation/main/loss'], 
                                  'epoch', file_name='loss.png')) 
        trainer.extend( 
            extensions.PlotReport( 
                ['main/accuracy', 'validation/main/accuracy'], 
                'epoch', file_name='accuracy.png')) 


    # Print selected entries of the log to stdout 
    # Here "main" refers to the target link of the "main" optimizer again, and 
    # "validation" refers to the default name of the Evaluator extension. 
    # Entries other than 'epoch' are reported by the Classifier link, called by 
    # either the updater or the evaluator. 
    trainer.extend(extensions.PrintReport( 
        ['epoch', 'main/loss', 'validation/main/loss', 
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time'])) 


    # Print a progress bar to stdout 
    trainer.extend(extensions.ProgressBar()) 


    if args.resume: 
        # Resume from a snapshot 
        chainer.serializers.load_npz(args.resume, trainer) 


    # Run the training 
    trainer.run() 

if __name__ == '__main__': 
    main() 

# パラメータの設定
# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 100
    
# 学習の繰り返し回数
n_epoch = 20
    
# 中間層の数
n_units = 1000

# モデル構築用データの準備    
# データセットのロード
iris  = load_iris()
    
# 学習用データ
iris.data  = iris.data.astype(np.float32)
    
# iris.target : 正解データ　（教師データ）
iris.target = iris.target.astype(np.int32)

# テストデータセットを訓練セットとテストセットに分割 (8:2)
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.2)

# モデルを構築
# 多層パーセプトロンモデルの設定
# 入力：150次元、出力3次元
model = FunctionSet(l1=F.Linear(150, n_units),
                    l2=F.Linear(n_units, n_units),
                    l3=F.Linear(n_units, 3))

# ニューラルネットワークの構造
def forward(x_data, y_data, train = True):
    x, t = Variable(x_data),Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)), train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    # 多クラス分類なので誤差関数としてソフトマックス関数の
    # 交差エントロピー関数を用いて誤差を導出
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# Optimizerの設定
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

# ミニバッチ学習
train_loss = []
train_acc = []
test_loss = []
test_acc = []

l1_W = []
l2_W = []
l3_W = []

# learning loop
for epoch in xrange(1, n_epoch+1):
    print ("epoch", epoch)
    
    # training
    # N個の順番をランダムに並び替える
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    # 0～Nまでのデータをバッチサイズごとに使用し学習
    for i in xrange(0, N, batchsize):
        x_batch = X_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]
        
        # 勾配を初期化
        optimizer.zero_grads()
        # 順伝播させ誤差と精度を算出
        loss, acc = forward(x_batch, y_batch)
        # 誤差逆伝播させ勾配を計算
        loss.backward()
        optimizer.update()
        
        train_loss.append(loss.data)
        train_acc.append(acc.data)
        sum_loss        += float(cuda.tocpu(loss.data)) * batchsize
        sum_accuracy    += float(cuda.tocpu(acc.data)) * batchsize
        
        # 訓練データ誤差と正解精度を表示
        print ("train mean loss={}, accuracy={}".format(sum_loss / N, sum_accuracy / N)
    
        # evaluation
        # テストデータで誤差と正解精度を算出し汎化性能を確認    
        # 勾配を初期化
        optimizer.zero_grads()
        # 順伝播させ誤差と精度を算出
        loss, acc = forward(x_batch, y_batch, train=False)
    
        test_loss.append(loss.data)
        test_acc.append(acc.data)
        sum_loss        += float(cuda.tocpu(loss.data)) * batchsize
        sum_accuracy    += float(cuda.tocpu(acc.data)) * batchsize
        
        # テストデータでの誤差と正解精度を表示
        print 'test mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test)
    
        # 学習したパラメータを保存
        l1_W.append(model.l1.W)
        l2_W.append(model.l2.W)
        l3_W.append(model.l3.W)
    
# 精度と誤差をグラフ表示
plt.figure(figsize=(8, 6))
plt.plot(range(len(train_acc)), train_acc)
plt.plot(range(len(test_acc)), test_acc)
plt.legend(["train_acc","test_acc"], loc=4)
plt.title("Accuracy of digit recognition.")
plt.plot()