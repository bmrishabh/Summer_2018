#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Initial conditions and fitness function
    for co-evolution simulations.
    @author: Saumil Shah'''

import numpy as np
from numba import jit

pa0, pb0 = np.arange(1,11), np.arange(1,11) #array of phenotypes
na0, nb0 = 10*np.ones(10), 10*np.ones(10)   #array of frequencies (uniform)

x, y =  np.meshgrid(pa0, pb0) #pre-interaction matrices of phenotypes

'''c[n] - coefficients of x^n          belongs to R '''

c = [0, 1]

@jit
def f(p):

    '''fitness as a function of phenotype.
    Make changes in the following expression.
    Following is equivalent to f(x) = c1*x + c0 '''

    return c[1]*np.power(p, 1) + c[0]