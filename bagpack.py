import numpy as np
from scipy import stats
import pandas as pd

def LPF(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*xlist[i] + b*ylist).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*ylist[i] + b*xlist).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def LNF(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*xlist[i] - b*ylist).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*ylist[i] - b*xlist).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def POUPF(b, xlist, ylist):
    a = 5.5; c = 0.66
    fecx = np.array([np.mean(np.array(((xlist[i]-a)**2)+c + b*(((ylist-a)**2)+c)).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(((ylist[i]-a)**2)+c + b*(((xlist-a)**2)+c)).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def POUNF(b, xlist, ylist):
    a = 5.5; c = 0.66
    fecx = np.array([np.mean(np.array(((xlist[i]-a)**2)+c - b*(((ylist-a)**2)+c)).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(((ylist[i]-a)**2)+c - b*(((xlist-a)**2)+c)).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def PODPF(b, xlist, ylist):
    a = 5.5; c = 21.25
    fecx = np.array([np.mean(np.array(-((xlist[i]-a)**2)+c + b*(-((ylist-a)**2)+c)).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(-((ylist[i]-a)**2)+c + b*(-((xlist-a)**2)+c)).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def PODNF(b, xlist, ylist):
    a = 5.5; c = 21.25
    fecx = np.array([np.mean(np.array(-((xlist[i]-a)**2)+c - b*(-((ylist-a)**2)+c)).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(-((ylist[i]-a)**2)+c - b*(-((xlist-a)**2)+c)).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def SODPF(b, xlist, ylist):
    a = 4; c = 20.5
    fecx = np.array([np.mean(np.array((xlist[i]*a*(1 - (xlist[i]/c))) + b*(ylist*a*(1 - (ylist/c)))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array((ylist[i]*a*(1 - (ylist[i]/c))) + b*(xlist*a*(1 - (xlist/c)))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def SODNF(b, xlist, ylist):
    a = 4; c = 20.5
    fecx = np.array([np.mean(np.array((xlist[i]*a*(1 - (xlist[i]/c))) - b*(ylist*a*(1 - (ylist/c)))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array((ylist[i]*a*(1 - (ylist[i]/c))) - b*(xlist*a*(1 - (xlist/c)))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def SOUPF(b, xlist, ylist):
    a = 0.186; c = 3.78
    fecx = np.array([np.mean(np.array(((xlist[i] - 1)*xlist[i]*a)+c + b*(((ylist - 1)*ylist*a))+c).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(((ylist[i] - 1)*ylist[i]*a)+c + b*(((xlist - 1)*xlist*a))+c).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def SOUNF(b, xlist, ylist):
    a = 0.186; c = 3.78
    fecx = np.array([np.mean(np.array(((xlist[i] - 1)*xlist[i]*a)+c - b*(((ylist - 1)*ylist*a))+c).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(((ylist[i] - 1)*ylist[i]*a)+c - b*(((xlist - 1)*xlist*a))+c).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def XYupPF(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*((xlist[i]*ylist)/(xlist[i]+ylist)) + b*((ylist[i]*xlist)/(ylist[i]+xlist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*((ylist[i]*xlist)/(ylist[i]+xlist)) + b*((xlist[i]*ylist)/(xlist[i]+ylist))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def XYupNF(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*((xlist[i]*ylist)/(xlist[i]+ylist)) - b*((ylist[i]*xlist)/(ylist[i]+xlist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*((ylist[i]*xlist)/(ylist[i]+xlist)) - b*((xlist[i]*ylist)/(xlist[i]+ylist))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def XYdownPF(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*((xlist[i]+ylist)/(xlist[i]*ylist)) + b*((ylist[i]+xlist)/(ylist[i]*xlist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*((ylist[i]+xlist)/(ylist[i]*xlist)) + b*((xlist[i]+ylist)/(xlist[i]*ylist))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def XYdowNF(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*((xlist[i]+ylist)/(xlist[i]*ylist)) - b*((ylist[i]+xlist)/(ylist[i]*xlist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*((ylist[i]+xlist)/(ylist[i]*xlist)) - b*((xlist[i]+ylist)/(xlist[i]*ylist))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def X_YsquarePF(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*(np.square(xlist[i]-ylist)) + b*(np.square(ylist-xlist[i]))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*(np.square(ylist[i]-xlist)) + b*(np.square(xlist-ylist[i]))).clip(0)) for i in range(len(xlist))])
    return fecx, fecy

def X_YsquareNF(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*(np.square(xlist[i]-ylist)) - b*(np.square(ylist-xlist[i]))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*(np.square(ylist[i]-xlist)) - b*(np.square(xlist-ylist[i]))).clip(0)) for i in range(len(xlist))])
    return fecx, fecy

def X_YsquareNOint(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*(np.square(xlist[i]-ylist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*(np.square(ylist[i]-xlist))).clip(0)) for i in range(len(xlist))])
    return fecx, fecy
    
def XplusYsquarePF(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*(np.square(xlist[i]+ylist)) + b*(np.square(ylist+xlist[i]))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*(np.square(ylist[i]+xlist)) + b*(np.square(xlist+ylist[i]))).clip(0)) for i in range(len(xlist))])
    return fecx, fecy
    
def XplusYsquareNF(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*(np.square(xlist[i]+ylist)) - b*(np.square(ylist+xlist[i]))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*(np.square(ylist[i]+xlist)) - b*(np.square(xlist+ylist[i]))).clip(0)) for i in range(len(xlist))])
    return fecx, fecy
    
def XplusYsquareNOint(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(a*((xlist[i]+ylist)**2)).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*((ylist[i]+xlist)**2)).clip(0)) for i in range(len(xlist))])
    return fecx, fecy

def XplusSWS(b, xlist, ylist):#Sum Whole Square
    a = 1
    fecx = np.array([np.mean(np.array(a*xlist[i] + b*(np.square(xlist[i]+ylist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*ylist[i] + b*(np.square(ylist[i]+xlist))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def XminusSWS(b, xlist, ylist):#Sum Whole Square
    a = 1
    fecx = np.array([np.mean(np.array(a*xlist[i] - b*(np.square(xlist[i]+ylist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*ylist[i] - b*(np.square(ylist[i]+xlist))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def XplusDWS(b, xlist, ylist):#Difference Whole Square
    a = 1
    fecx = np.array([np.mean(np.array(a*xlist[i] + b*(np.square(xlist[i]-ylist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*ylist[i] + b*(np.square(ylist[i]-xlist))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def XminusDWS(b, xlist, ylist):#Difference Whole Square
    a = 1
    fecx = np.array([np.mean(np.array(a*xlist[i] + b*(np.square(xlist[i]-ylist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(a*ylist[i] + b*(np.square(ylist[i]-xlist))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def aXplusbYsquare(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(np.square(a*xlist[i] + b*(ylist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(np.square(a*ylist[i] + b*(xlist))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy

def aXminusbYsquare(b, xlist, ylist):
    a = 1
    fecx = np.array([np.mean(np.array(np.square(a*xlist[i] - b*(ylist))).clip(0)) for i in range(len(xlist))])
    fecy = np.array([np.mean(np.array(np.square(a*ylist[i] - b*(xlist))).clip(0)) for i in range(len(ylist))])
    return fecx, fecy
    
def Shannon(xlist, ylist):
    Ex = stats.entropy(np.unique(xlist, return_counts = True)[1])
    Ey = stats.entropy(np.unique(ylist, return_counts = True)[1])
    return Ex, Ey

def Newlist(popsize, probx, proby, xlist, ylist):
    xnum = np.random.multinomial(popsize, probx)
    ynum = np.random.multinomial(popsize, proby)
    xlist = np.repeat(xlist, xnum)
    ylist = np.repeat(ylist, ynum)
    return xlist, ylist

def TTF(Sx, Sy, gens):
    if 0 in Sx:
        TTFx = np.where(Sx == 0)[0][0] + 1 # +1 because indexing starts from 0
    else:
        TTFx = gens+1
    if 0 in Sy:
        TTFy = np.where(Sy == 0)[0][0] + 1
    else:
        TTFy = gens+1
    return TTFx, TTFy

def Dataframe(avgx, avgy, Sx, Sy, avgfecx, avgfecy, timex, timey, gens):
    # d is a dictionary where all the final averages and std are stored
    ## axis = 1 implies the operation is done row wise i.e. for a gen across all reps
    d = {'avgxpg':np.mean(avgx, axis = 1), 'stdxpg':np.std(avgx, axis = 1),
         'avgypg':np.mean(avgy, axis = 1), 'stdypg':np.std(avgy, axis = 1),
         'Sxpg':np.mean(Sx, axis = 1), 'stdSxpg':np.std(Sx, axis = 1),
         'Sypg':np.mean(Sy, axis = 1), 'stdSypg':np.std(Sy, axis = 1),
         'avgfecxpg':np.mean(avgfecx, axis = 1), 'stdfecxpg':np.std(avgfecx, axis = 1),
         'avgfecypg':np.mean(avgfecy, axis = 1), 'stdfecypg':np.std(avgfecy, axis = 1),
         'generations':np.arange(1,gens+1,1), 'empty':np.array([]),
         'tfix_x':timex, 'tfix_y':timey}

    # df is a pandas data frame, transpose gives data column wise, 
    df = pd.DataFrame.from_dict(data = d, orient = 'index').transpose()
    # defining order of the rows in the data frame
    df = df[['generations', 'avgxpg', 'stdxpg', 'Sxpg', 'stdSxpg', 'avgfecxpg', 'stdfecxpg', 'tfix_x',\
             'empty', 'generations', 'avgypg', 'stdypg', 'Sypg', 'stdSypg', 'avgfecypg', 'stdfecypg', 'tfix_y']]
    return df

def Dataframeb(brange, timex, timey):
    d = {'b':brange, 'empty':np.array([]),
         'tfix_x':np.mean(timex, axis = 1), 'tfix_y':np.mean(timey, axis = 1),
         'stdtfix_x':np.std(timex, axis =1), 'stdtfix_y':np.std(timey, axis = 1)}

    # df is a pandas data frame, transpose gives data column wise,
    df = pd.DataFrame.from_dict(data = d, orient = 'index').transpose()
    # defining order of the rows in the data frame
    df = df[['b', 'tfix_x', 'stdtfix_x',\
             'empty', 'b', 'tfix_y', 'stdtfix_y']]
    return df

def Normpop(start, stop, steps, popsize):
    mu = (stop - start)/2; std = (mu - start)/3
    prob = stats.norm.cdf(np.arange(start,stop,steps)+0.5, loc = mu, scale = std)-\
           stats.norm.cdf(np.arange(start,stop,steps)-0.5, loc = mu, scale = std)
    prob = prob/sum(prob)
    xlist = np.random.choice(np.arange(start,stop,steps), size = popsize, p = prob) # initial pop x in normal dist
    ylist = np.random.choice(np.arange(start,stop,steps), size = popsize, p = prob) # initial pop y in normal dist
    return xlist, ylist

