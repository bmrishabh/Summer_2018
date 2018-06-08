import time # to track the time required
start = time.time() # getting start time
import numpy as np # for mean, std, array, repeats, multinomial
import pandas as pd # to create data frame and write in excel
from scipy import stats # for shannon entropy
import itertools # to concatenate lists inside a list

gens = 200 # No. of generations
reps = 20 # No. of replicates
popsize = 100 # No. of individuals in a population (fixed)
brange = np.arange(0, 1.05, 0.05) # b from 0 to 1.05 with steps 0.05

# creating arrays of 'len(brange)' rows and 'reps' columns
timex, timey = np.array([np.zeros([len(brange),reps]) for i in range(2)])
for i in range(len(brange)): # loop for changing b
    b = brange[i] # taking values from brange

    # creating arrays of 'gens' rows and 'reps' columns
    Sx, Sy= np.array([np.zeros([gens, reps]) for i in range(2)])
    
    for r in range(reps): # loop for reps
        xlist = np.repeat(np.arange(1,11,1), 10) # initial pop x
        ylist = np.repeat(np.arange(1,11,1), 10) # initial pop y

        for g in range(gens): # loop for generations
            a = 1 # a=intrinsic fitness coef

            # fecx, fecy are fitness functions for x and y
            ## np.array.clip(0)-> values below 0 put to 0
            fecx = np.array([np.mean(np.array(a*xlist[i] + b*ylist).clip(0)) for i in range(len(xlist))]).clip(0)
            fecy = np.array([np.mean(np.array(a*ylist[i] + b*xlist).clip(0)) for i in range(len(ylist))]).clip(0)

            # Probability of each phenotype value in pop x and y
            ## max(1., sum(fecx)) gives division s.t. sum(probx) is 1
            probx = fecx/max(1., sum(fecx))
            proby = fecy/max(1., sum(fecy))

            # stats.entropy(prob list) gives shannon entropy
            ## set(fecx) gives all the elements in fecx only ones
            if sum(list(set(fecx))) != 0:
                Sx[g,r] = stats.entropy(np.array(list(set(fecx)))/max(1., sum(list(set(fecx)))))
            else:
                Sx[g,r] = 0
            if sum(list(set(fecy))) != 0:
                Sy[g,r] = stats.entropy(np.array(list(set(fecy)))/max(1., sum(list(set(fecy)))))
            else:
                Sy[g,r] = 0

            # xnum, ynum are random multinomial output for probx and proby
            ## Sum(xnum) amd Sum(ynum) is equal to popsize
            xnum = np.random.multinomial(popsize, probx)
            ynum = np.random.multinomial(popsize, proby)

            # Getting new xlist and ylist using xnum and ynum            
            xlist = np.repeat(xlist,xnum)
            ylist = np.repeat(ylist,ynum)
        # Finding 0 in entropy array, getting its index, which is the gen of fixation
        ## If pop is not fixed then time to fixation is kept as gens+1
        if 0 in Sx[:,r]:
            timex[i,r] = list(Sx[:,r]).index(0) + 1 # +1 because indexing starts from 0
        else:
            timex[i,r] = gens+1
        if 0 in Sy[:,r]:
            timey[i,r] = list(Sy[:,r]).index(0) + 1
        else:
            timey[i,r] = gens+1
    print(b)

# d is a dictionary where all the final averages and std are stored
## axis = 1 implies the operation is done row wise i.e. for given b across all reps
d = {'b':brange, 'empty':np.array([]),
     'tfix_x':np.mean(timex, axis = 1), 'tfix_y':np.mean(timey, axis = 1),
     'stdtfix_x':np.std(timex, axis =1), 'stdtfix_y':np.std(timey, axis = 1)}

# df is a pandas data frame, transpose gives data column wise,
df = pd.DataFrame.from_dict(data = d, orient = 'index').transpose()
# defining order of the rows in the data frame
df = df[['b', 'tfix_x', 'stdtfix_x',\
         'empty', 'b', 'tfix_y', 'stdtfix_y']]

# Creating new xlsx file to be written
writer = pd.ExcelWriter('New code changing b.xlsx')
# Writing the data in the xlsx file
df.to_excel(writer, 'Sheet1',index = False, merge_cells = False)
writer.save() # Saving the file

end = time.time() # getting end time
print(end - start, "sec") # gives time required to execute the code
