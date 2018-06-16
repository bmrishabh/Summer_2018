#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Function library for co-evolution simulations.
    @author: Saumil Shah'''

import numpy as np
import scipy.stats
from numba import jit
from matplotlib import rc
from matplotlib import pyplot as plt

from coevo import *

#changes fontsize in plots
rc('font', size=22)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)

def muan(p):

    '''Finds the type of interaction
    given a vale of parameter b.'''

    tem = {1:'mut', 0:'nil', -1:'ant'}
    return tem[np.sign(p)]

@jit
def norm(p):

    '''Normalizes the given array.'''

    return (p / sum(p))

@jit
def shan(p):

    '''Calculates shannon index
    of the given frequency array.'''

    return scipy.stats.entropy(p)

def ftau(p, maxgen):

    '''Finds fixation time,
    returns generation when shannon index
    vanishes for the first time.'''

    if np.argmin(p > 0) == 0:
        return maxgen+1
    else:
        return np.argmin(p > 0)

@jit
def efec(naj, nbj, a, b, maxpop):

    '''Evaluates effective fecundity array from interactions.
    Following is equivalent to f(x) = a(c1*x + c0) + b(c1*y + c0)'''
    
    efa = ( a*f(pa0) + b*np.average(f(pb0), weights=nbj) ).clip(0) * naj
    efb = ( a*f(pb0) + b*np.average(f(pa0), weights=naj) ).clip(0) * nbj
    return efa/maxpop, efb/maxpop

@jit
def grow(naj, nbj, a, b, maxpop):

    '''Draws offspring population sample.'''

    efa, efb = efec(naj, nbj, a, b, maxpop)
    return np.random.multinomial(maxpop, pvals=norm(efa)), np.random.multinomial(maxpop, pvals=norm(efb))

@jit
def gaug(naj, nbj, a, b, maxpop):

    '''Calculates required population properties.'''

    efa, efb = efec(naj, nbj, a, b, maxpop)
    eopa, eopb = np.average(pa0, weights=naj), np.average(pb0, weights=nbj)
    eofa, eofb = np.average(efa, weights=naj), np.average(efb, weights=nbj)
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
    plt.savefig('plots/{}_b{}_phe.png'.format(muan(b), np.abs(b)))
    plt.close()

def eofe(avg, std, fta, ftb, b):

    '''Plots mean fecundity vs generations.'''

    plt.figure(figsize=(12,8))
    plt.errorbar(avg.index, avg['f(x)'], std['f(x)'], label=fta)
    plt.errorbar(avg.index, avg['f(y)'], std['f(y)'], label=ftb)
    plt.xlabel('Generations')
    plt.ylabel('Mean Fecundity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/{}_b{}_fec.png'.format(muan(b), np.abs(b)))
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
    plt.savefig('plots/{}_b{}_sha.png'.format(muan(b), np.abs(b)))
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
    
def fec1(a, b):

    '''Plots intrinsic fecundity vs phenotypes.'''

    plt.figure(figsize=(12,8))
    plt.plot(pa0, a*f(pa0), label='Intrinsic')
    plt.xlabel('Phenotypes')
    plt.ylabel('Intrinsic Fecundity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/b{}_fec1.png'.format(np.abs(b)))
    plt.close()

def fec2(b):

    '''Plots interaction fecundity vs phenotypes.'''

    plt.figure(figsize=(12,8))
    plt.plot(pa0,  b*f(pa0), label='+ve b')
    plt.plot(pa0, -b*f(pa0), label='-ve b')
    plt.xlabel('Phenotypes')
    plt.ylabel('Interaction Fecundity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/b{}_fec2.png'.format(np.abs(b)))
    plt.close()
    
def fec3(a, b):

    '''Plots composite fecundity vs phenotypes.'''

    plt.figure(figsize=(12,8))
    plt.plot(pa0, a*f(pa0) + b*f(pa0), label='+ve b')
    plt.plot(pa0, a*f(pa0), label='nil b')
    plt.plot(pa0, a*f(pa0) - b*f(pa0), label='-ve b')
    plt.xlabel('Phenotypes')
    plt.ylabel('Composite Fecundity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/b{}_fec3.png'.format(np.abs(b)))
    plt.close()