#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Initial conditions and fitness function
    for co-evolution simulations.
    @author: Saumil Shah'''

import numpy as np
from numba import jit
from scipy import stats as ss

maxgen = 200 #number of generations
maxpop = 100 #number of individuals
maxrep = 50  #number of replicate runs

pa0, pb0 = np.arange(1,11), np.arange(1,11) #array of phenotypes

pu0, pl0 = pa0 + 0.5, pa0 - 0.5
prb = ss.norm(5.5,1.5).cdf(pu0) - ss.norm(5.5,1.5).cdf(pl0)
prb = prb / sum(prb)

x, y =  np.meshgrid(pa0, pb0) #pre-interaction matrices of phenotypes

'''c[n] - coefficients of x^n          belongs to R '''

@jit
def f(x,y):

    '''fitness as a function of phenotype.
    Make changes in the following expression.
    Following is equivalent to f(x) = c1*x + c0 '''

    rgb = {'lin':x,
           'qou':np.power(x-5.5,2) + 0.66,
           'qod':-np.power(x-5.5,2) + 21.25,
           'sou':0.186*np.multiply(x,x-1) + 3.78,
           'sod':0.186*np.multiply(x,1- (x/20.5)),
           'som':np.divide(x+y, np.multiply(x,y)),
           'mos':np.divide(np.multiply(x,y), x+y),
           'dws':np.power(x-y,2),
           'sws':np.power(x+y,2)}
    
    return rgb['sws']