#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:25:24 2020

@author: nachshon
"""

import pandas as pd
import numpy as np
import glob
import openpyxl as ex
from sklearn.preprocessing import RobustScaler

US = ['CSAplusUS', 'CSBplusUS']

sub_list = pd.read_csv('/media/Data/Lab_Projects/RCF/sublistandgroup.csv')
stim_list = pd.read_csv('/media/Data/Lab_Projects/RCF/stimlist.csv')
wb = ex.load_workbook(filename='/media/Data/Lab_Projects/RCF/stimlist.xlsx', data_only=True)

df = pd.DataFrame()
glober = '/home/nachshon/Documents/RCF/SCR/RCF*.txt'

for scr in glob.glob(glober):
    sub = scr.split("RCF")[2].split("_")[0]
    df_temp = pd.read_table(scr)
    df_temp['sub']=sub
    df_temp['group'] = sub_list[sub][0]
    df_temp['list'] = sub_list[sub][1]
    df_temp['type'] = 0
    for i in range(2,43):
        df_temp['type'][i-2] = wb[df_temp['list'][1]].cell(i,3).value
    
    df_temp = df_temp[~df_temp['type'].isin(US)]
    
    cda = np.array(df_temp['CDA.SCR'])
    cda = cda.reshape(-1,1)
    transformer = RobustScaler().fit(cda)
    
    df_temp['transform'] = transformer.transform(cda)
    
    df = pd.concat([df, df_temp])
    
df.to_csv('/media/Data/Lab_Projects/RCF/normalizedSCR.csv', index=False)
