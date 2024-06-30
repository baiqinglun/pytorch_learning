import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

X1 = torch.rand(10000,1) # 输入特征1
X2 = torch.rand(10000,1) # 输入特征2
X3 = torch.rand(10000,1) # 输入特征3
Y1 = ( (X1+X2+X3)<1 ).float() # 输出特征1
Y2 = ( (1<(X1+X2+X3)) & ((X1+X2+X3)<2) ).float() # 输出特征2
Y3 = ( (X1+X2+X3)>2 ).float() # 输出特征3
Data = torch.cat([X1,X2,X3,Y1,Y2,Y3],axis=1) # 整合数据集
Data = Data.to('cuda:0') # 把数据集搬到GPU 上
Data.shape

print(X1)