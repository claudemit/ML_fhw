#-*- coding:utf-8 -*-
__author__ = 'claudemit13@gmail.com'
__copyright__ = '<claudemit>'

import numpy as np
import random
from math import sqrt

max_iter = 50
try_num = 2000

def ini_data(data_fn):
    res=np.loadtxt(data_fn)
    res=np.insert(res,0,1,axis=1)
    return res

data =ini_data('ntumlone 2Fhw1 2Fhw1_18_train.dat')
test =ini_data('ntumlone 2Fhw1 2Fhw1_18_test.dat')
len=data.shape[0]

def pocket_train():

    def _calc_false(vec):
        res = 0
        mistakes=[]
        for i in xrange(len):
            f=data[i][:5]
            t = sum(vec * f)
            if np.sign(data[i][5])!= np.sign(t):
                res += 1
                mistakes.append(i)
        return res,mistakes

    weights = np.array([0,0,0,0,0])
    w = np.array([0,0,0,0,0])
    w_err,mis=_calc_false(w)

    iter_num = 0

    while iter_num < max_iter:
        row=data[random.choice(mis)]
        feature = row[:5]
        train_result =sum(weights * feature)
        result = row[5]
        if result * train_result > 0:
            continue
        weights = weights + result * feature
        temp_err,mis=_calc_false(weights)
        iter_num+=1

        if w_err>temp_err:
            w=weights
            w_err=temp_err

    return w,weights

if __name__ == '__main__':

    win,w0=pocket_train()
    print win,w0

    a=1
    aver_w=np.array([0,0,0,0,0])
    aver_ws=np.array([0,0,0,0,0])
    for i in range(try_num):
        random.Random(i)
        w,ws=pocket_train()
        aver_w+=w
        aver_ws+=ws
    print aver_w/try_num,aver_ws/try_num
    # print sqrt(np.dot(aver_w.T,aver_w))
