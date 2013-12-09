#-*- coding:utf-8 -*-

import numpy as np
import random
from math import sqrt

def ini_data(data_fn):
    res=np.loadtxt(data_fn)
    res=np.insert(res,0,1,axis=1)
    return res

data=ini_data('ntumlone 2Fhw1 2Fhw1_15_train.dat')
len=data.shape[0]

def train():
    weights = np.array([0,0,0,0,0])
    update=1
    flag = 1
    while flag:
        flag = 0
        for item in data:
            feature = item[:5]
            train_result = sum(weights * feature)
            result = item[5]
            if result * train_result > 0:
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
            f=row[:5]
            tr=np.dot(w.T,f)
            rrnd=row[5]
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
    a=0.5
    aver_up=0.0
    aver_w=0
    for i in range(num):
        random.Random(i)
        rndlist=list(range(len))
        random.shuffle(rndlist)
        wf,up=rndtrain(rndlist,a)
        aver_up+=up
        aver_w+=wf
    print aver_up/num,aver_w/num
