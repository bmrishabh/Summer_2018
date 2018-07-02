#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Function library for co-evolution simulations.
    @author: Saumil Shah'''

import numpy as np
from numba import jit
from matplotlib import rc
from scipy import stats as ss
from matplotlib import pyplot as plt

from coevo import *

#changes fontsize in plots
rc('font', size=28)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

def muan(p):

    '''Finds the type of interaction
    given a vale of parameter b.'''

    tem = {1:'mut', 0:'nil', -1:'ant'}
    return tem[np.sign(p)]

@jit
def norm(p):

    '''Normalizes the given array.'''
    if np.sum(p) != 0:
        return (p / np.sum(p))
    else:
        return p

@jit
def shan(p):

    '''Calculates shannon index
    of the given population.'''

    return ss.entropy(np.unique(p, return_counts=True)[1])

def ftau(p):

    '''Finds fixation time,
    returns generation when shannon index
    vanishes for the first time.'''

    if np.argmin(p > 0) == 0:
        return maxgen+1
    else:
        return np.argmin(p > 0)

@jit
def efec(paj, pbj, a, b):

    '''Evaluates effective fecundity array from interactions.
    Following is equivalent to a*f(x,y) + b*f(y,x)'''

    x, y =  np.meshgrid(paj, pbj)

    efa = ( np.average(a*f(x,y)+b*f(y,x), axis=0) ).clip(0)
    efb = ( np.average(a*f(y,x)+b*f(x,y), axis=1) ).clip(0)

    return efa, efb

@jit
def grow(paj, pbj, a, b):

    '''Draws offspring population sample.'''

    efa, efb = efec(paj, pbj, a, b)

    noa = np.random.multinomial(maxpop, norm(efa)).clip(0)
    nob = np.random.multinomial(maxpop, norm(efb)).clip(0)

    return np.repeat(paj, noa), np.repeat(pbj, nob)

@jit
def gaug(paj, pbj, a, b):

    '''Calculates required population properties.'''

    efa, efb = efec(paj, pbj, a, b)
    eopa, eopb = np.average(paj), np.average(paj)
    eofa, eofb = np.average(efa), np.average(efb)
    sopa, sopb = shan(paj), shan(pbj)
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

def prof(cmy, a, b):

    '''Plots three fecundity vs phenotypes,
    and fixation time vs parameter b figures.'''

    x, y = np.meshgrid(phen, phen)
    fec0 = np.average(a*f(x,y), axis=0)
    fecp = np.average(a*f(x,y) + b*f(y,x), axis=0)
    fecn = np.average(a*f(x,y) - b*f(y,x), axis=0).clip(0)
    
    plt.figure(figsize=(32,18))
    
    plt.subplot(221)
    plt.plot(phen, fec0, label='Intrinsic')
    plt.xlabel('Phenotypes')
    plt.ylabel('Intrinsic Fecundity')
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(223)
    plt.plot(phen, fecp - fec0, label='+ve b')
    plt.plot(phen, fecn - fec0, label='-ve b')
    plt.xlabel('Phenotypes')
    plt.ylabel('Interaction Fecundity')
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(222)
    plt.plot(phen, fecp, label='+ve b')
    plt.plot(phen, fec0, label='nil b')
    plt.plot(phen, fecn, label='-ve b')
    plt.xlabel('Phenotypes')
    plt.ylabel('Composite Fecundity')
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(224)
    plt.errorbar(np.abs(cmy['b']), cmy['efta'], cmy['dfta'], label='A')
    plt.errorbar(np.abs(cmy['b']), cmy['eftb'], cmy['dftb'], label='B')
    plt.xlabel('b')
    plt.ylabel('Fixation time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/b{}_prof.png'.format(np.abs(b)))
    plt.close()

''' Finds probability of phenotypes
from normal distribution around 5.5 units
with standard deviation of 1.5 units.'''
  
upperlim, lowerlim = phen + 0.5, phen - 0.5
prob = ss.norm(5.5,1.5).cdf(upperlim) - ss.norm(5.5,1.5).cdf(lowerlim)
prob = norm(prob)