#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Function library for co-evolution simulations.
    Only make changes in the box marked by hyphens.
    @author: Saumil Shah'''

import numpy as np
import scipy.stats
from numba import jit
from matplotlib import rc
from matplotlib import pyplot as plt

#changes fontsize in plots
rc('font', size=22)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)

def muan(b):

    '''Finds the type of interaction
    given a vale of parameter b.'''

    tem = {1:'mut', 0:'nil', -1:'ant'}
    return tem[np.sign(b)]

@jit
def norm(p):

    '''Normalizes the given array.'''

    return (p / sum(p))

@jit
def shan(p):

    '''Calculates shannon index
    of the given frequency array.'''

    return scipy.stats.entropy(p)

def ftau(ser, maxgen):

    '''Finds fixation time,
    returns generation when shannon index
    vanishes for the first time.'''

    if np.argmin(ser > 0) == 0:
        return maxgen+1
    else:
        return np.argmin(ser > 0)

#-----------------------------------------------------------------------------#

@jit
def f(sfj, c):

    '''fitness as a function of phenotype.
    Make changes in the expression following return.
    Following is equivalent to f(x) = c1*x + c0 '''

    return c[1]*sfj + c[0]

@jit
def efit(naj, nbj, pa0, pb0, a, b, c):

    '''Evaluates effective fitness array from interactions.
    Following is equivalent to f(x) = a(c1*x + c0) + b(c1*y + c0)'''

    sfa, sfb = np.meshgrid(pa0, pb0)
    efa = ( np.average(a*f(sfa) + b*f(sfb), axis=0) ).clip(0) * naj
    efb = ( np.average(a*f(sfb) + b*f(sfa), axis=1) ).clip(0) * nbj
    return efa, efb

#-----------------------------------------------------------------------------#

@jit
def grow(naj, nbj, pa0, pb0, a, b, c, maxpop):

    '''Draws offspring population sample.'''

    efa, efb = efit(naj, nbj, pa0, pb0, a, b, c)
    return np.random.multinomial(maxpop, pvals=norm(efa)), \
            np.random.multinomial(maxpop, pvals=norm(efb))

@jit
def gaug(naj, nbj, pa0, pb0, a, b, c, maxpop):

    '''Calculates required population properties.'''

    efa, efb = efit(naj, nbj, pa0, pb0, a, b, c)
    eopa, eopb = np.average(pa0, weights=naj), \
                    np.average(pb0, weights=nbj)
    eofa, eofb = np.average(efa, weights=naj)/maxpop, \
                    np.average(efb, weights=nbj)/maxpop
    sopa, sopb = shan(naj), shan(nbj)
    return [eopa, eopb, eofa, eofb, sopa, sopb]

def eoph(avg, std, fta, ftb, b):

    '''Plots mean phenotype vs generations.'''

    plt.figure(figsize=(12,8))
    plt.errorbar(avg.index, avg['e(x)'], std['e(x)'], label=fta)
    plt.errorbar(avg.index, avg['e(y)'], std['e(y)'], label=ftb)
    plt.xlabel('Generations')
    plt.ylabel('Mean Phenotype')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/b{}_{}_phe.png'.format(np.abs(b), muan(b)))
    plt.close()

def eofi(avg, std, fta, ftb, b):

    '''Plots mean fitness vs generations.'''

    plt.figure(figsize=(12,8))
    plt.errorbar(avg.index, avg['f(x)'], std['f(x)'], label=fta)
    plt.errorbar(avg.index, avg['f(y)'], std['f(y)'], label=ftb)
    plt.xlabel('Generations')
    plt.ylabel('Mean Fitness')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/b{}_{}_fit.png'.format(np.abs(b), muan(b)))
    plt.close()

def eosh(avg, std, fta, ftb, b):

    '''Plots mean shannon index vs generations.'''

    plt.figure(figsize=(12,8))
    plt.errorbar(avg.index, avg['s(x)'], std['s(x)'], label=fta)
    plt.errorbar(avg.index, avg['s(y)'], std['s(y)'], label=ftb)
    plt.xlabel('Generations')
    plt.ylabel('Mean Shannon Index')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/b{}_{}_sha.png'.format(np.abs(b), muan(b)))
    plt.close()

def tren(cmy):

    '''Plots fixation time vs parameter b.'''

    plt.figure(figsize=(12,8))
    plt.errorbar(cmy['b'], cmy['efta'], cmy['dfta'], label='A')
    plt.errorbar(cmy['b'], cmy['eftb'], cmy['dftb'], label='B')
    plt.xlabel('b')
    plt.ylabel('Fixation time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/trend.png')
    plt.close()