import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import sklearn.cluster as skc


correctcount=0
correctpic=0
for i in range(30):
	cnt=0
	loc1=[]
	loc2=[]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	GG=np.loadtxt('./raw/'+str(i)+'LowQ.txt',delimiter=' ')
	X=GG[:,0]
	Y=GG[:,1]
	Z=GG[:,2]
	for j in range(19):
		loc1.append(X[j]-1+(Y[j]-1)*10)


	ax.scatter(X,Y,marker='o',s=100)


	QQ=np.loadtxt('./vali/'+str(i)+'.txt',delimiter=' ')
	XX=QQ[:,0]
	YY=QQ[:,1]
	ax.scatter(XX,YY,marker='x',s=100)
	for j in range(len(XX)):
		temp=XX[j]-1+(YY[j]-1)*10
		if temp in loc1:
			cnt+=1
	if cnt/19 > 0.6 :
		correctpic+=1
	correctcount+=cnt
	plt.show()

print(correctcount/570)
print(correctpic)



