#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:49:09 2019

Reference: https://github.com/poldracklab/ds003-post-fMRIPrep-analysis/blob/master/workflows.py

@author: rj299
"""
import nipype.interfaces.io as nio  # Data i/o
from nipype.interfaces import spm
from nipype import Node, Workflow, MapNode
import nipype.interfaces.utility as util # utility
from nipype import SelectFiles
import os

from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('/home/nachshon/Documents/MATLAB/spm12/') # set default SPM12 path in my computer. 

#%% Gourp analysis - based on SPM - should consider the fsl Randomize option (other script)
# OneSampleTTestDesign - creates one sample T-Test Design
onesamplettestdes = Node(spm.OneSampleTTestDesign(),
                         name="onesampttestdes")

# EstimateModel - estimates the model
level2estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                      name="level2estimate")

# EstimateContrast - estimates group contrast
level2conestimate = Node(spm.EstimateContrast(group_contrast=True),
                         name="level2conestimate")
cont1 = ['Group', 'T', ['mean'], [1]]
level2conestimate.inputs.contrasts = [cont1]

# Which contrasts to use for the 2nd-level analysis
contrast_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004']

subject_list =['030' , '1005', '1072', '1074', '1099', '1205', '1206',
 '1210', '1212', '1216', '1218', '1220', '1221', '1223', '1237', 
 '1245', '1247', '1254', '1258', '1266',  '1269', '1271',
 '1272', '1280', '1290', '1291', '1301', '1303', '1309', '1312',
 '1319', '1320', '1326', '1337', '1338', '1343',
 '1346', '1350',  '1376',
 '1378', '1445', '1449']
# '1268', , '1340' '1345', '1347', '1357', '1359', '1362', '1374', '1379', '1384', '1388', '1389', '1392', '1393', '1440', '1444', , '1457', '1460'
# Threshold - thresholds contrasts
level2thresh = Node(spm.Threshold(contrast_index=1,
                              use_topo_fdr=False,
                              use_fwe_correction=True, # here we can use fwe or fdr
                              extent_threshold=10,
                              height_threshold= 0.05,
                              extent_fdr_p_threshold = 0.05,
                              height_threshold_type='p-value'),
                              
                                   name="level2thresh")

 #Infosource - a function free node to iterate over the list of subject names
infosource = Node(util.IdentityInterface(fields=['contrast_id', 'subject_id']),
                  name="infosource")

infosource.iterables = [('contrast_id', contrast_list)]
infosource.inputs.subject_id = subject_list

# SelectFiles - to grab the data (alternative to DataGrabber)
templates = {'cons': os.path.join('/media/Data/work/RCF9/imaging/Sink_resp/1stLevel/_subject_id_{subject_id}/', 
                         '{contrast_id}.nii')}

selectfiles = MapNode(SelectFiles(templates,
                               base_directory='/media/Data/work/RCF9/work',
                               sort_filelist=True),
                   name="selectfiles", 
                   iterfield = ['subject_id'])

datasink = Node(nio.DataSink(base_directory='/media/Data/work/RCF9/imaging/Sink_resp'),
                name="datasink")


l2analysis = Workflow(name='l2spm')

l2analysis.base_dir = '/media/Data/work/RCF9/work'

l2analysis.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                               ('subject_id', 'subject_id')]),

                    (selectfiles, onesamplettestdes, [('cons', 'in_files')]),
                    
                    (onesamplettestdes, level2estimate, [('spm_mat_file',
                                                          'spm_mat_file')]),
                    (level2estimate, level2conestimate, [('spm_mat_file',
                                                          'spm_mat_file'),
                                                         ('beta_images',
                                                          'beta_images'),
                                                         ('residual_image',
                                                          'residual_image')]),
                    (level2conestimate, level2thresh, [('spm_mat_file',
                                                        'spm_mat_file'),
                                                       ('spmT_images',
                                                        'stat_image'),
                                                       ]),
                    (level2conestimate, datasink, [('spm_mat_file',
                        '2ndLevel.@spm_mat'),
                       ('spmT_images',
                        '2ndLevel.@T'),
                       ('con_images',
                        '2ndLevel.@con')]),
                    (level2thresh, datasink, [('thresholded_map',
                                               '2ndLevel.@threshold')]),
                                                        ])
#%%                                                     
l2analysis.run('MultiProc', plugin_args={'n_procs': 14})
