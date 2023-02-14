import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import sklearn.cluster as skc

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
GG=np.loadtxt('result1.txt',delimiter=' ')
X=[]
Y=[]
Z=[]
for i in range(512):
	if GG[i]==1:
		ZZ=i//64 +1
		YY=(i-64*(ZZ-1))//8 +1
		XX=i%8 +1
		X.append(XX)
		Y.append(YY)
		Z.append(ZZ)
ax.scatter(X,Y,Z,marker='o',s=100)


QQ=np.loadtxt('30.txt',delimiter=' ')
XXX=[]
YYY=[]
ZZZ=[]
for i in range(512):
	if QQ[i]==1:
		ZZ=i//64 +1
		YY=(i-64*(ZZ-1))//8 +1
		XX=i%8 +1
		XXX.append(XX)
		YYY.append(YY)
		ZZZ.append(ZZ)
ax.scatter(XXX,YYY,ZZZ,marker='x',s=100)
plt.show()




