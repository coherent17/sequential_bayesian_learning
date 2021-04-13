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
    return dataX

def shuffle(dataX,dataT):
    data_temp=np.c_[dataT,dataX]
    np.random.shuffle(data_temp)
    dataT=data_temp[:,0]
    dataX=np.delete(data_temp,[0],axis=1)
    return dataX,dataT

def sigmoidal(dataX):
    for i in range(0,len(dataX)):
        dataX[i]=math.exp(dataX[i])/(1+math.exp(dataX[i]))
    return dataX

def linear_regression(X,Y):
    w=np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T,X)),X.T),Y)
    return w

def hypothesis(w,X):
    return np.matmul(w,np.transpose(X))

dataX,dataT=shuffle(dataX,dataT)

#N=5
dataX_sig=np.copy(dataX)
dataX_sig=normalization(dataX_sig)
dataX_sig=sigmoidal(dataX_sig)
dataX_sig=np.c_[np.array([1]*len(dataX_sig)),dataX_sig]
#take 5 points to train
w=linear_regression(dataX_sig[0:5],dataT[0:5])

#generate the testing dataset of x
#need to do data preprocessing as same as the given x data
x=np.linspace(0,2,200)
x_sig=np.copy(x)
x_sig=normalization(x_sig)
x_sig=sigmoidal(x_sig)
x_sig=np.c_[np.array([1]*len(x_sig)),x_sig]
y=hypothesis(w,x_sig)

#visualize
plt.plot(x,y,color="r")
plt.plot(dataX[:],dataT[:],'bo')
plt.show()