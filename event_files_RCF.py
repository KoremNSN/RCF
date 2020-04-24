#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:20:20 2020

@author: nachshon

input - 1 csv with subject events onset and duration
output - n files (num of subjects) named sub-X.csv with trail type
onset and duration
without shocked trials.
"""

import pandas as pd
import os


os.chdir('/media/Data/work/RCF/event_files')
data = pd.read_csv('/home/nachshon/Documents/RA_PTSD/Reconsolidation/events/events.csv')
sub = "0"
file = sub +".csv"
f = ""
title = 'trial_type,onset,duration\n'
for line in data.iterrows():
    if str(line[1][0]) != sub:
        sub = str(line[1][0])
        file = "sub-" + sub + ".csv"
        f = open(file, 'w')
        f.write(title)
    #if line[1][1][-3:] != "sUS":
    row = line[1][1] + "," + str(line[1][2]) + "," + str(line[1][3]) + "\n"
    f.write(row)
