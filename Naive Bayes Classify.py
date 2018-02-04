import numpy as np
import csv as csv
import os
import matplotlib.pyplot as plt 


path='C:\\Users\\Jessie\\SkyDrive\\2017Fall\\Machine Learning\\HW4'
os.chdir(path)

  
def BernoulliNB(X,Y,Test,m,phat):
    n,d = X.shape
    D=np.hstack((X,Y))
    pi0= np.zeros(d)
    pi1= np.zeros(d)
    pz=np.count_nonzero(Y==1)/n
    predict=np.zeros(Test.shape[0])
    for i in range(d):
        pi0[i]=(np.count_nonzero(D[D[:,22]==0][:,i] == 1)+m*phat)/(n+m)
        pi1[i]=(np.count_nonzero(D[D[:,22]==1][:,i] == 1)+m*phat)/(n+m)
    for i in range(Test.shape[0]):
        z0=(1-pz)*np.prod(np.power(pi0,Test[i,:])*np.power(1-pi0,1-Test[i,:]))
        z1=pz*np.prod(np.power(pi1,Test[i,:])*np.power(1-pi1,1-Test[i,:]))
        if z0>z1:
            predict[i]=0
        else:
            predict[i]=1
    return predict
            
        
TrainX=[]
with open('SpectTrainData.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        TrainX.append(row)
TrainX=np.array(TrainX,dtype='int')

TrainY=[]
with open('SpectTrainLabels.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        TrainY.append(row)
TrainY=np.array(TrainY,dtype='int')

TestX=[]
with open('SpectTestData.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        TestX.append(row)
TestX=np.array(TestX,dtype='int')

TestY=[]
with open('SpectTestLabels.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        TestY.append(row)
TestY=np.array(TestY,dtype='int')

predict=BernoulliNB(TrainX,TrainY,TestX,2,0.5)

Error=np.array(predict-np.ravel(TestY),dtype='int')

Errorrate=1-np.count_nonzero(Error == 0)/len(Error)
print(Errorrate)

def f(t):
    return (np.sqrt(t/(1-t)))*(1-t)+ (np.sqrt((1-t)/t))*(t)-1
t1 = np.arange(0.0, 1, 0.01)


plt.figure(1)

plt.plot(t1, f(t1), 'k')

plt.show()

