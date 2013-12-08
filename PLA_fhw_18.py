#-*- coding:utf-8 -*-
__author__ = 'claudemit13@gmail.com'
__copyright__ = '<claudemit>'

import numpy as np
import random
from math import sqrt

data =np.loadtxt('ntumlone 2Fhw1 2Fhw1_18_train.dat')
test =np.loadtxt('ntumlone 2Fhw1 2Fhw1_18_test.dat')
len=data.shape[0]
max_iter=50
try_num=2000

def pocket_train():
    ###############
    def _calc_false(vec):
        res = 0
        for item in data:
            t = sum(vec * item[:4])
            if np.sign(item[4]) != np.sign(t):
                res += 1
        return res
        ###############

    weights = np.array([0,0,0,0,0])
    w=weights.copy
    w_err=_calc_false(w)

    for i in xrange(max_iter):
        row=random.choice(data)
        feature = row[:4]
        feature=np.insert(feature,0,values=1)
        train_result =sum(weights * feature)
        result = row[4]
        if result * train_result > 0:
            continue
        weights = weights + result * feature
        temp_err=_calc_false(weights)
        if w_err>temp_err:
            w=weights

    return w,weights

if __name__ == '__main__':
    win,w50=pocket_train()
    print win,w50

    a=1
    aver_w=np.array([0,0,0,0,0])
    aver_ws=np.array([0,0,0,0,0])
    for i in range(try_num):
        random.seed(i)
        w,ws=pocket_train()
        aver_w+=w
        aver_ws+=ws
    print aver_w/try_num,aver_ws/try_num
    print sqrt(np.dot(aver_w.T,aver_w))