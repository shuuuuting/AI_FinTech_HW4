# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import Counter
from sklearn import datasets
iris = datasets.load_iris()
feature = iris.data[0:150,:]
target = iris.target[0:150]

def Kmeans(sample,K,maxiter): #maxiter:最多經歷的iteration數
    N = sample.shape[0]
    D = sample.shape[1]
    C = np.zeros((K,D)) #中心點
    L = np.zeros((N,1)) #點的label
    L1 = np.zeros((N,1)) 
    dist = np.zeros((N,K)) #distance
    idx = random.sample(range(N),K)
    C = sample[idx,:]
    iteration = 0
    while(iteration<maxiter):
        for i in range(K):
            dist[:,i] = np.sum((sample-np.tile(C[i,:],(N,1)))**2,1) #tile:把C[i,:]重複N*1遍
        L1 = np.argmin(dist,1)
        if(iteration>0 and np.array_equal(L,L1)):
            print(iteration)
            break
        L=L1
        for i in range(K): #K個中心點
            idx = np.nonzero(L==i)[0]
            if(len(idx)>0):
                C[i,:] = np.mean(sample[idx,:],0)
        iteration+=1
#        G1 = sample[L==0,:]
#        G2 = sample[L==1,:]
#        G3 = sample[L==2,:]
#        plt.plot(G1[:,0],G1[:,1],'r.',G2[:,0],G2[:,1],'g.',G3[:,0],G3[:,1],'b.',C[:,0],C[:,1],'kx')
#        plot.show()
    #wicd = np.sum(np.sqrt(np.sum((sample-C[L,:])**2,1)))#每個點跟群中心的距離(within class distance)和越小越好
    return C,L

def KNN(test,train,target,k):
    N = train.shape[0]
    dist = np.sum((np.tile(test,(N,1))-train)**2,1)      
    idx = sorted(range(len(dist)),key=lambda i:dist[i])[0:k] #依距離由小到大取key值 if dist=[3,1,2],sort=[1,2,0]
    #idx挑出哪k個點最近
    return Counter(target[idx]).most_common(1)[0][0] #回傳0,1,2最多的那一個,(1)代表取1個
'''
G1 = np.random.normal(0,1,(5000,2)) #G1: mean=[4 4], std=[1,1]
G1[:,0] = G1[:,0]+4
G1[:,1] = G1[:,1]+4

G2 = np.random.normal(0,1,(3000,2)) #G2:mean=[0,-3], std=[1,3]
G2[:,1] = G2[:,1]*3-3

G3 = np.random.normal(0,1,(2000,2)) #G3: mean=[-4,6], std=[1,4], ckwise
G3[:,1] = G3[:,1]*4
c45 = math.cos(-45/180*math.pi) #rotation = 45 degree counterclockwise
s45 = math.sin(-45/180*math.pi)
R = np.array([[c45,-s45],[s45,c45]])
G3 = G3.dot(R)
G3[:,0] = G3[:,0]-4
G3[:,1] = G3[:,1]+6
G=np.append(G1,G2,axis=0)
G = np.append(G,G3,axis=0)
plt.plot(G[:,0],G[:,1],'.')
'''

G = feature
C,L = Kmeans(G,3,1000)
G1 = G[L==0,:]
G2 = G[L==1,:]
G3 = G[L==2,:]
#plt.plot(G1[:,0],G1[:,1],'r.',G2[:,0],G2[:,1],'g.',G3[:,0],G3[:,1],'b.',C[:,0],C[:,1],'kx')
wicd = np.sum(np.sqrt(np.sum((G-C[L,:])**2,1)))
print('pure wicd =',wicd)

#standard score
GA = (G-np.tile(np.mean(G,0),(G.shape[0],1)))/np.tile(np.std(G,0),(G.shape[0],1))
C,L = Kmeans(GA,3,1000)
G1 = G[L==0,:]
G2 = G[L==1,:]
G3 = G[L==2,:]
wicd = np.sum(np.sqrt(np.sum((G-C[L,:])**2,1)))
print('standard score wicd =',wicd)
#scaling
GB = (G-np.tile(np.min(G,0),(G.shape[0],1)))/np.tile((np.max(G,0)-np.min(G,0)),(G.shape[0],1))
C,L = Kmeans(GB,3,1000)
G1 = G[L==0,:]
G2 = G[L==1,:]
G3 = G[L==2,:]
wicd = np.sum(np.sqrt(np.sum((G-C[L,:])**2,1)))
print('scaling wicd =',wicd)

for k in range(1,11):
    label=[]
    count = 0
    confusion = np.zeros(shape=(3,3))
    for i in range(150):
        label.append(KNN(feature[i],np.r_[feature[:i],feature[i+1:]],np.r_[target[:i],target[i+1:]],k))
    for j in range(len(target)):
        if (target[j]==label[j]): count+=1
        confusion[target[j]][label[j]]+=1
    print(k,'NN confusion matrix')
    print('    0','  1','  2')
    print('0',confusion[0])
    print('1',confusion[1])
    print('2',confusion[2])
    print()
        