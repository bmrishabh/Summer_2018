#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''multiple b value co-evolution simulations.
    @author: Saumil Shah'''

import gc
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from coevo import *
from stuff import *

def sims(i):
    
    '''Simulates interactions and reproductions.'''
    
    gc.collect()
    df=np.array([])
    ft=np.array([])
    
    pa0 = np.repeat(phen, np.random.multinomial(maxgen, prob))
    pb0 = np.repeat(phen, np.random.multinomial(maxgen, prob))
            
    paj, pbj = pa0, pb0
    df = np.array([gaug(paj, pbj, a, b)])

    for g in range(maxgen):
        paj, pbj = grow(paj, pbj, a, b)
        df = np.vstack((df,[gaug(paj, pbj, a, b)]))

    ft = np.array([ftau(df[:,4]), ftau(df[:,5])])
      
    return [df, ft]

'''Following are fitness parameters.
    a - self interaction strength       belongs to [0,1]
    b - non-self interaction strength   belongs to [0,1]
    +/- ve values of b mean mutualism/antagonism.
    b values are being iterated in a loop below.'''

a = 1
cmy = []

for b in np.arange(-0.602t,1.05,0.05).round(2):
    print(' b =', b)
    if __name__ == '__main__':
        p=Pool(cpu_count()-1)
        l = p.map(sims, [i for i in range(maxrep)])
        p.close()
        p.join()

    l=np.array(l)
    col = ['e(x)', 'e(y)', 'f(x)', 'f(y)', 's(x)', 's(y)']
    avg = pd.DataFrame(np.average(np.vstack(l[:,0]).reshape((50,201,6)), axis=0), columns=col).round(2)
    std = pd.DataFrame(    np.std(np.vstack(l[:,0]).reshape((50,201,6)), axis=0), columns=col).round(2)
    
    eoft = np.average(np.vstack(l[:,1]), axis=0).round(2)
    doft =     np.std(np.vstack(l[:,1]), axis=0).round(2)
    
    cmy = cmy + [[b, eoft[0], doft[0], eoft[1], doft[1]]]

cmy = pd.DataFrame(cmy, columns=['b', 'efta', 'dfta', 'eftb', 'dftb'])
prof(cmy, 1, 0.5)
print(' Done!')