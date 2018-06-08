import gc
import numpy as np
import pandas as pd
from coevo import grow, gaug, ftau, eoph, eofi, eosh

maxgen = 200
maxpop = 100
maxrep = 20
a, b, c = 1, 0.5, [1,0]

df = np.zeros((maxrep,maxgen+1,6))
ft = np.zeros((maxrep,2))

ifa, ifb = np.arange(1,11), np.arange(1,11)
na0, nb0 = 10*np.ones(10), 10*np.ones(10)

for r in range(maxrep):    
    gc.collect()    
    naj, nbj = na0, nb0
    df[r,0] = np.array(gaug(naj, nbj, ifa, ifb, a, b, c, maxpop))    
    
    for g in range(maxgen):
        naj, nbj = grow(naj, nbj, ifa, ifb, a, b, c, maxpop)        
        df[r,g+1] = np.array(gaug(naj, nbj, ifa, ifb, a, b, c, maxpop))        
    ft[r] = [ftau(df[r,:,4], maxgen), ftau(df[r,:,5], maxgen)]

col = ['e(x)', 'e(y)', 'f(x)', 'f(y)', 's(x)', 's(y)']
avg = pd.DataFrame(np.average(df, axis=0), columns=col).round(2)
std = pd.DataFrame(    np.std(df, axis=0), columns=col).round(2)

eoft = np.average(ft, axis=0).round(2)
doft =     np.std(ft, axis=0).round(2)

fta = ' F.t. for a: {} ± {} gens'.format(eoft[0], doft[0])
ftb = ' F.t. for b: {} ± {} gens'.format(eoft[1], doft[1])

eoph(avg, std, fta, ftb)
eofi(avg, std, fta, ftb)
eosh(avg, std, fta, ftb)