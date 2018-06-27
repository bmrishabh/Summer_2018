#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''single b value co-evolution simulations.
    @author: Saumil Shah'''

import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from coevo import *
from stuff import *

'''Following are fitness parameters.
    a - self interaction strength       belongs to [0,1]
    b - non-self interaction strength   belongs to [0,1]
    +/- ve values of b mean mutualism/antagonism.'''

a, b = 1, 0.5

df = np.zeros((maxrep,maxgen+1,6)) #array to store sample moments
ft = np.zeros((maxrep,2))          #array to store fixation time

for r in range(maxrep):    
    gc.collect()    
    naj, nbj = na0, nb0
    df[r,0] = np.array(gaug(naj, nbj,a, b))

    for g in range(maxgen):
        naj, nbj = grow(naj, nbj, a, b)        
        df[r,g+1] = np.array(gaug(naj, nbj, a, b))

    ft[r] = [ftau(df[r,:,4]), ftau(df[r,:,5])]

col = ['e(x)', 'e(y)', 'f(x)', 'f(y)', 's(x)', 's(y)']
avg = pd.DataFrame(np.average(df, axis=0), columns=col).round(2)
std = pd.DataFrame(    np.std(df, axis=0), columns=col).round(2)

eoft = np.average(ft, axis=0).round(2)
doft =     np.std(ft, axis=0).round(2)

fta = ' F.t. for A: {} ± {} gens'.format(eoft[0], doft[0])
ftb = ' F.t. for B: {} ± {} gens'.format(eoft[1], doft[1])

eoph(avg, std, fta, ftb, b)
eofe(avg, std, fta, ftb, b)
eosh(avg, std, fta, ftb, b)

fec1(a, b)
fec2(a, b)
fec3(a, b)
print(' Done!')