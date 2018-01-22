# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:01:34 2017

@author: Matsuura
"課題＿決定木ver"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn


# 文字列は既に置換済
# unknownは一律「99」で変換。
df_bank = pd.read_csv('C:/wk/ucl/bank-additional_forPython.csv')


