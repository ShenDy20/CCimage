import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
df = pd.read_csv(r'C:\Users\25345\Desktop\final project\clust\C400.txt', names=['X','Y','Z','E','x','y','z','e'])
AA = df['X']
print(type(AA))  # <class 'pandas.core.series.Series'>
a = np.array(AA)
print(type(a))  # <class 'numpy.ndarray'>
a = a.tolist()
print(type(a))  # <class 'list'>
a=list(map(float,a))
a = torch.unsqueeze(torch.FloatTensor(a), dim=1)
print(type(a))  # <class 'torch.Tensor'>
print(a.shape)  # torch.Size([97, 1])


