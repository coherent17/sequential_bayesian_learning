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
    dataS=1/(1+np.exp(-1*dataX))
    return dataS

def linear_regression(X,Y):
    beta=1
    m_0=np.zeros(len(X))
    S_0_inv=(10**-6)*np.eye(len(X)) #(5*5)
    S_N_inv=S_0_inv+beta*np.dot(X.T,X)
    S_N=np.linalg.pinv(S_N_inv)
    m_N=S_N@(np.dot(S_0_inv,m_0)+beta*np.dot(X.T,Y))
    return m_N

def hypothesis(m_N,X):
    return np.matmul(m_N.T,X)

dataX,dataT=shuffle(dataX,dataT)

#N=5
dataX_train=np.copy(dataX)
dataX_train=normalization(dataX_train)
dataX_train=sigmoidal(dataX_train)

#take 5 points to train
m_N=linear_regression(dataX_train[0:5],dataT[0:5])
print(m_N)
print(m_N.shape)

#generate the testing dataset of x
#need to do data preprocessing as same as the given x data
# x=np.random.uniform(low=0,high=2,size=(200,1))
# x_sig=np.copy(x)
# x_sig=normalization(x_sig)
# x_sig=sigmoidal(x_sig)
# y_sig=hypothesis(m_N,x_sig)
# print(y_sig)

# #visualize
# plt.plot(x,y_sig)
# plt.plot(dataX[:],dataT[:],'bo')
# plt.show()

