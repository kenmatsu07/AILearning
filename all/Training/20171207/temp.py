# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
X_train,X_test,y_train,y_test = train_test_split(
        iris_dataset['data'],iris_dataset['target'],random_state=0)

# =============================================================================
# 1.7.2～1.7.3

# print ("X_train shape: {}".format(X_train.shape))
# print ("y_train shape: {}".format(y_train.shape))
# print ("X_test shape: {}".format(X_test.shape))
# print ("y_test shape: {}".format(y_test.shape))
# 
# 
iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
# 
# grr = pd.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',
#                         hist_kwds={'bins' : 20},s=60,alpha=.8, cmap=mglearn.cm3)
# =============================================================================

# =============================================================================
# 1.7.4～1.7.5
 
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
# 
# 
X_new = np.array([[5,2.9,1,0.2]])
# print ("X_new.shape: {}".format(X_new.shape))
# 
# prediction = knn.predict(X_new)
# print ("Prediction: {}".format(prediction))
# print ("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
# =============================================================================
 
 # 1.7.6～1.7.5
 
y_pred = knn.predict(X_test)
print ("Test set predictions:\n {}".format(y_pred))

print ("Test set score: {:.2f}".format(np.mean(y_pred==y_test)))