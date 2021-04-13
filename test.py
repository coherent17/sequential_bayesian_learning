import numpy as np
import matplotlib.pyplot as plt
import math

#feature data
dataX=np.genfromtxt("x.csv",delimiter=',')
#target data
dataT=np.genfromtxt("t.csv",delimiter=',')

def normalization(dataX):
    for i in range(0,len(dataX)):
        dataX[i]=(dataX[i]-2*i/3)/0.1

def shuffle(dataX,dataT):
    data_temp=np.c_[dataT,dataX]
    np.random.shuffle(data_temp)
    dataT=data_temp[:,0]
    dataX=np.delete(data_temp,[0],axis=1)

def sigmoidal(dataX):
    for i in range(0,len(dataX)):
        dataX[i]=math.exp(dataX[i])/(1+math.exp(dataX[i]))
    return dataX

def linear_regression(X,Y):
    w=np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T,X)),X.T),Y)
    return w


shuffle(dataX,dataT)
#N=5
dataX_sig=dataX
normalization(dataX_sig)
dataX_sig=sigmoidal(dataX_sig)
dataX_sig=np.c_[np.array([1]*len(dataX_sig)),dataX_sig]
#take 5 points to train
w=linear_regression(dataX_sig[0:5],dataT[0:5])

#visualize:
x=np.linspace(0,2,20000)
y=w[0]+w[1]*x
plt.plot(x,y)
plt.show()