import numpy as np
import matplotlib.pyplot as plt

#data preprocesing:
#feature data
dataX=np.genfromtxt("x.csv",delimiter=',')
#target data
dataT=np.genfromtxt("t.csv",delimiter=',')

print(dataX)

print(dataT)