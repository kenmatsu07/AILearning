# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 16:25:05 2017

@author: Matsuura
""""chainerなしのディープラーニング　（バッチ勾配降下）"

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

class Config:
    nn_input_dim = 2                # インプット層の次元数
    nn_output_dim = 2               # アウトプット層の次元数
    # ハイパーパラメータ (数値は一般的に使われる値を採用)
    epsilon = 0.01                  # 学習率
    reg_lambda = 0.01               # 重み

# データを生成する
def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

# グラフ設定関数を呼び出す
def visualize(X, y, model):
    plot_dicision_boundary(lambda x: predict(model, x), X, y)
    plt.title("Logistic Regression")
    # 結果をファイル出力する。
    filename = "DeepLearning.jpg"
    plt.savefig(filename)

# グラフ設定用の関数
def plot_dicision_boundary(pred_func, X, y):
    # 最小値、最大値から目盛を定める
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    
    # グラフを描画
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show

# 全損失関数を計算するためのHelper function
def calculate_loss(model, X, y):
    num_examples = len(X)        # 学習用データサイズ
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 予測算出のためのForward Propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # 損失関数を計算
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # 損失関数にregulatization termを与える
    data_loss += Config.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# 勾配を計算 (返却値：0、1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 順勾配伝播
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
   
    # テスト用乱数を生成
    num_examples = len(X)        # 学習用データサイズ
    np.random.seed(0)
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.nn_output_dim))
    
    model = {}
    
    # 確率的勾配降下法で深層学習を行う
    for i in range(0, num_passes):
        
        # 順勾配伝播
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores,axis=1, keepdims=True)
    
        # 逆勾配伝播
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # ハイパーパラメータを用いて計算
        dW2 += Config.reg_lambda * W2
        dW1 += Config.reg_lambda * W1
        
        # ハイパーパラメータを更新
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2
        
        # 最適化されたハイパーパラメターを格納
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" %(i, calculate_loss(model, X, y))
            
    return model

def classify(X, y): 
    # clf = linear_model.LogisticRegressionCV() 
    # clf.fit(X, y) 
    # return clf 
    pass 

def main(): 
    X, y = generate_data() 
    model = build_model(X, y, 3, print_loss=True) 
    visualize(X, y, model) 

if __name__ == "__main__": 
    main()
