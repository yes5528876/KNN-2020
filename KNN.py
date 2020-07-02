# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:40:58 2020

@author: yes55
"""

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

#讀取資料開始
path='C:/Users/yes55/Desktop/data/Train/'  ##file reading
files=os.listdir(path)
train=[]
for file in files:
    p=path+file
    #資料預處理
    img=cv2.resize(cv2.imread(p),(500,500),interpolation=cv2.INTER_CUBIC)
    img=img.reshape(750000)
    train.append(img)
    
ground=pd.read_csv('C:/Users/yes55/Desktop/data/train.csv')
y_train=np.array(ground.iloc[0:1000,1])

    
path='C:/Users/yes55/Desktop/data/Test/'  ##file reading
files=os.listdir(path)
test=[]
for file in files:
    p=path+file
    #資料預處理
    img=cv2.resize(cv2.imread(p),(500,500),interpolation=cv2.INTER_CUBIC)
    img=img.reshape(750000)
    test.append(img)
    

#PCA    
pca = PCA(n_components = 10)
pca.fit(train)
PCA(copy=True, iterated_power='auto', n_components=10, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)

x_train = pca.transform(train)
x_test=pca.transform(test)

#KNN
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)





