#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:09:25 2020

@author: nachshon
"""


import glob

sub_list = []
glober = '/media/Data/Lab_Projects/RCF/work/work/l1spm_resp/_subject_id_*/svScrub/percentScrub.txt'

for f in glob.glob(glober):
    with open(f) as df:
        print("sub-"+f.split('/')[8].split("_")[-1] + " " + df.readlines()[0])