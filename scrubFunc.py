#!/usr/bin/env python

def scrub(file, thr=0.3):
    import pandas as pd
    import numpy as np
    # takes fmriPrep's FD and threshold (usually 0.3) and returns scrubbed column
    regressor = pd.read_csv(file, sep=r'\s+')
    x = np.zeros(len(regressor))
    for i, fd in enumerate(regressor.framewise_displacement.fillna(0)):
        
        if fd>thr:
            x[i] = 1
            x[i-1] = 1
            x[i+1] = 1
        else:
            continue
    per= np.array([sum(x)/len(x)])
    np.savetxt('percentScrub.txt',per) # saving percentage of scrubbing to txt file so we can track it
    regressor['scrub'] = x
    return regressor
    #regressor.to_csv(file, sep='\t', index=False)
