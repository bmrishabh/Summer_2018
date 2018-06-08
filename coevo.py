import numpy as np
import scipy.stats
from numba import jit
import matplotlib

matplotlib.rcParams.update({'font.size': 22})
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
plt = matplotlib.pyplot

@jit
def norm(p):
    return (p / sum(p))

@jit
def shan(p):
    return scipy.stats.entropy(p)

def ftau(ser, maxgen):
    if np.argmin(ser > 0) == 0:
        return maxgen+1
    else:
        return np.argmin(ser > 0)

@jit
def efit(naj, nbj, ifa, ifb, a, b, c):
    sfa, sfb = np.meshgrid(ifa, ifb)
    efa = ( np.average(a*c[0]*sfa + c[1] + b*c[0]*sfb, axis=0) ).clip(0) * naj
    efb = ( np.average(a*c[0]*sfb + c[1] + b*c[0]*sfa, axis=1) ).clip(0) * nbj
    return efa, efb

@jit
def grow(naj, nbj, ifa, ifb, a, b, c, maxpop):
    efa, efb = efit(naj, nbj, ifa, ifb, a, b, c)
    return np.random.multinomial(maxpop, pvals=norm(efa)), np.random.multinomial(maxpop, pvals=norm(efb))

@jit
def gaug(naj, nbj, ifa, ifb, a, b, c, maxpop):
    efa, efb = efit(naj, nbj, ifa, ifb, a, b, c)
    eopa, eopb = np.average(ifa, weights=naj), np.average(ifb, weights=nbj)
    eofa, eofb = np.average(efa, weights=naj)/maxpop, np.average(efb, weights=nbj)/maxpop
    sopa, sopb = shan(naj), shan(nbj)
    return [eopa, eopb, eofa, eofb, sopa, sopb]        

def eoph(avg, std, fta, ftb):
    plt.figure(figsize=(12,8))
    plt.errorbar(avg.index, avg['e(x)'], std['e(x)'], label=fta)
    plt.errorbar(avg.index, avg['e(y)'], std['e(y)'], label=ftb)
    plt.xlabel('Generations')
    plt.ylabel('Mean Phenotype')
    plt.legend()
    plt.show()
    plt.close()
    
def eofi(avg, std, fta, ftb):
    plt.figure(figsize=(12,8))
    plt.errorbar(avg.index, avg['f(x)'], std['f(x)'], label=fta)
    plt.errorbar(avg.index, avg['f(y)'], std['f(y)'], label=ftb)
    plt.xlabel('Generations')
    plt.ylabel('Mean Fecundity')
    plt.legend()
    plt.show()
    plt.close()

def eosh(avg, std, fta, ftb):
    plt.figure(figsize=(12,8))
    plt.errorbar(avg.index, avg['s(x)'], std['s(x)'], label=fta)
    plt.errorbar(avg.index, avg['s(y)'], std['s(y)'], label=ftb)
    plt.xlabel('Generations')
    plt.ylabel('Mean Shannon Index')
    plt.legend()    
    plt.show()
    plt.close()