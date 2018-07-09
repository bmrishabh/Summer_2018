import time # to track the time required
start = time.time() # getting start time
import numpy as np # for mean, std, array, repeats, multinomial
import pandas as pd # to create data frame and write in excel
from scipy import stats # for shannon entropy
from bagpack import *

gens = 200 # No. of generations
reps = 50 # No. of replicates
popsize = 100 # No. of individuals in a population (fixed)

# creating arrays of 'gens' rows and 'reps' columns
avgx, avgy, Sx, Sy, avgfecx, avgfecy = np.array([np.zeros([gens, reps]) for i in range(6)])
# creating arrays of 'reps' length for time to fixation
timex, timey = np.array([np.zeros(reps) for i in range(2)])

for r in range(reps): # loop for reps
    xlist, ylist = Normpop(1, 11, 1, popsize)
    
    for g in range(gens): # loop for generations
        b = 1 # a=intrinsic fitness coef; b=interaction coef
        
        # fecx, fecy are fitness functions for x and y
        ## np.array.clip(0)-> values below 0 put to 0
        fecx, fecy = SODPF(b, xlist, ylist)
        
        # Probability of each phenotype value in pop x and y
        ## max(1., sum(fecx)) gives division s.t. sum(probx) is 1
        probx = fecx/max(1., sum(fecx))
        proby = fecy/max(1., sum(fecy))

        # Filling avg phenotype in 'gth' generation and 'rth' repeat [row,col]
        avgx[g,r] = np.mean(xlist)
        avgy[g,r] = np.mean(ylist)

        # stats.entropy(prob list) gives shannon entropy
        ## set(fecx) gives all the elements in fecx only ones
        Sx[g,r], Sy[g,r] = Shannon(xlist, ylist)

        # Filling avg fitness in 'gth' gen and 'rth' rep
        avgfecx[g,r] = np.mean(fecx)
        avgfecy[g,r] = np.mean(fecy)

        # xnum, ynum are random multinomial output for probx and proby
        ## Sum(xnum) amd Sum(ynum) is equal to popsize
        xlist, ylist = Newlist(popsize, probx, proby, xlist, ylist)

    # Finding 0 in entropy array, getting its index, which is the gen of fixation
    ## If pop is not fixed then time to fixation is kept as gens+1
    timex[r], timey[r] = TTF(Sx[:, r], Sy[:, r], gens)
    print(r)

# d is a dictionary where all the final averages and std are stored
## axis = 1 implies the operation is done row wise i.e. for a gen across all reps
df = Dataframe(avgx, avgy, Sx, Sy, avgfecx, avgfecy, timex, timey, gens)
# Creating new xlsx file to be written
writer = pd.ExcelWriter('New code file.xlsx')
# Writing the data in the xlsx file
df.to_excel(writer, 'Sheet1',index = False, merge_cells = False)
writer.save() # Saving the file

end = time.time() # getting end time
print(end - start, "sec") # gives time required to execute the code
