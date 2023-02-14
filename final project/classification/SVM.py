from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchvision.models as models
import numpy as np
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
with open('./train_label.txt', 'r') as f:
    train = f.readlines()
with open('./test_label.txt', 'r') as g:
    test = g.readlines()
train_data = np.zeros((len(train), 4096))
train_lable =np.array(train,dtype=np.float32)

test_data = np.zeros((len(test), 4096))
test_lable =np.array(test,dtype=np.float32)

for i in range(len(train)):
    labels = []
    label=float(train[i])
    img_path='./figures/'+str(i+1)+'.png'
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = np.array(img).reshape(1,4096)[0]
    train_data[i]=img
    
for i in range(len(test)):
    img_path='./test/'+str(i+1)+'.png'
    labels = []
    label=float(test[i])
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = np.array(img).reshape(1,4096)[0]
    test_data[i]=img





# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(train_data,train_lable.astype('int'))
score_rbf = clf_rbf.score(test_data,test_lable.astype('int'))
print("The score of rbf is : %f"%score_rbf)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(train_data,train_lable.astype('int'))
score_linear = clf_linear.score(test_data,test_lable.astype('int'))
print("The score of linear is : %f"%score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(train_data,train_lable.astype('int'))
score_poly = clf_poly.score(test_data,test_lable.astype('int'))
print("The score of poly is : %f"%score_poly)
