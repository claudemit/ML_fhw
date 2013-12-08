#-*- coding:utf-8 -*-
# https://class.coursera.org/ntumlone-001/forum/thread?thread_id=95
import numpy as np
import random
from math import sqrt

data =np.loadtxt('ntumlone 2Fhw1 2Fhw1_15_train.dat')
len=data.shape[0]

def train():
    weights = np.array([0,0,0,0,0])
    update=1
    flag = 1
    while flag:
        flag = 0
        for item in data:
            feature = item[:4]
            feature=np.insert(feature,0,values=1)
            train_result = sum(weights * feature)
            result = item[4]
            if result * train_result > 0:    #不用纠错，忽略本次for循环中剩余代码，continue跳出本次循环；break跳出整个for循环
                continue
            weights = weights + result * feature
            update+=1
            flag = 1
    return weights,update

def rndtrain(rnd,a):
    w=np.array([0,0,0,0,0])
    uprnd=1
    flag1=1
    while flag1:
        flag1=0
        for i in rnd:
            row=data[i]
            f=row[:4]
            f=np.insert(f,0,values=1)
            tr=np.dot(w.T,f)
            rrnd=row[4]
            if np.sign(rrnd)==np.sign(tr):
                continue
            w=w+rrnd*f*a
            uprnd+=1
            flag1=1
    return sqrt(np.dot(w.T,w)),uprnd


if __name__ == '__main__':
    weights,update = train()
    print update,sqrt(np.dot(weights.T,weights))

    num=2000
    a=1
    aver_up=0.0
    aver_w=0
    for i in range(num):
        random.seed(i)
        rndlist=list(range(len))
        random.shuffle(rndlist)
        wf,up=rndtrain(rndlist,a)
        aver_up+=up
        aver_w+=wf
    print aver_up/num,aver_w/num