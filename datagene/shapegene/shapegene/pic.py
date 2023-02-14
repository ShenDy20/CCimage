import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import sklearn.cluster as skc

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
GG=np.loadtxt('500HighQ.txt',delimiter=' ')
X=GG[:,0]
Y=GG[:,1]
Z=GG[:,2]
DATA=np.array([[X[0],Y[0]]])
print(DATA)
for i in range(len(X)-1):
    temp=np.array([[X[i+1],Y[i+1]]])
    DATA=np.append(DATA,temp,axis=0)

print(DATA)
eps=2
mi=5
db=skc.DBSCAN(eps=eps,min_samples=mi).fit(DATA)
label = db.labels_
for j in range(len(label)):
    label[j]+=1

MAX=np.argmax(np.bincount(label))
XX=np.array([])
YY=np.array([])
ZZ=np.array([])

for k in range(len(label)):
    if label[k]==MAX:

        XX=np.append(XX,X[k])
        YY=np.append(YY,Y[k])
        ZZ=np.append(ZZ,Z[k])




ax.scatter(X,Y,Z,marker='o',s=100)
plt.show()
