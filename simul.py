#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''single b value co-evolution simulations.
    @author: Saumil Shah'''

import gc
import numpy as np
import pandas as pd
from coevo import grow, gaug, ftau, eoph, eofi, eosh

maxgen = 200 #number of generations
maxpop = 100 #number of individuals
maxrep = 20  #number of replicate runs

'''Following are fitness parameters.
    a - self interaction strength       belongs to [0,1]
    b - non-self interaction strength   belongs to [0,1]
    +/- ve values of b mean mutualism/antagonism.
    c[n] - coefficients of x^n          belongs to R '''

a, b, c = 1, 0.5, [0, 1]

df = np.zeros((maxrep,maxgen+1,6)) #array to store sample moments
ft = np.zeros((maxrep,2))          #array to store fixation time

pa0, pb0 = np.arange(1,11), np.arange(1,11) #array of phenotypes
na0, nb0 = 10*np.ones(10), 10*np.ones(10)   #array of frequencies

for r in range(maxrep):    
    gc.collect()    
    naj, nbj = na0, nb0
    df[r,0] = np.array(gaug(naj, nbj, pa0, pb0, a, b, c, maxpop))    

    for g in range(maxgen):
        naj, nbj = grow(naj, nbj, pa0, pb0, a, b, c, maxpop)        
        df[r,g+1] = np.array(gaug(naj, nbj, pa0, pb0, a, b, c, maxpop))

    ft[r] = [ftau(df[r,:,4], maxgen), ftau(df[r,:,5], maxgen)]

col = ['e(x)', 'e(y)', 'f(x)', 'f(y)', 's(x)', 's(y)']
avg = pd.DataFrame(np.average(df, axis=0), columns=col).round(2)
std = pd.DataFrame(    np.std(df, axis=0), columns=col).round(2)

eoft = np.average(ft, axis=0).round(2)
doft =     np.std(ft, axis=0).round(2)

fta = ' F.t. for A: {} ± {} gens'.format(eoft[0], doft[0])
ftb = ' F.t. for B: {} ± {} gens'.format(eoft[1], doft[1])

eoph(avg, std, fta, ftb, b)
eofi(avg, std, fta, ftb, b)
eosh(avg, std, fta, ftb, b)

print(' Done!')