#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Created on Wed Dec  4 14:29:06 2019

@author: Or Duek
1st level analysis using FSL output
In this one we smooth using SUSAN, which takes longer. 
"""

from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range

import os  # system functions

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as model  # model generation
#import nipype.algorithms.rapidart as ra  # artifact detection
from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth
from nipype.interfaces.utility import Function
"""
Preliminaries
-------------

Setup any package specific configuration. The output file format for FSL
routines is being set to compressed NIFTI.
"""

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
"""
Setting up workflows
--------------------

In this tutorial we will be setting up a hierarchical workflow for fsl
analysis. This will demonstrate how pre-defined workflows can be setup and
shared across users, projects and labs.
"""
#%%
data_dir = os.path.abspath('/media/Data/RCF_output/fmriprep')
output_dir = '/media/Data/work/RCF_or'
fwhm = 6
tr = 1
removeTR = 9#Number of TR's to remove before initiating the analysis
lastTR = 496
thr = 0.3 # scrubbing threshold
#%%


#%% Methods 
def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0, removeTR=4, lastTR=496, thr=0.3):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scrubFunc import scrub
    from nipype.interfaces.base.support import Bunch
    # Process the events file
    events = pd.read_csv(events_file)
    bunch_fields = ['onsets', 'durations', 'amplitudes']
    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]
    out_motion = Path('motion.par').resolve()
    regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    regress_data = scrub(regressors_file, thr)
    np.savetxt(out_motion, regress_data[motion_columns].values[removeTR:lastTR+removeTR,], '%g')
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))
    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']
    runinfo = Bunch(
        scans=in_file,
        conditions=list(set(events.trial_type.values)),
        **{k: [] for k in bunch_fields})
    for condition in runinfo.conditions:
        event = events[events.trial_type.str.match(condition)]
        runinfo.onsets.append(np.round(event.onset.values-removeTR, 3).tolist()) # added -removeTR to align to the onsets after removing X number of TRs from the scan
        runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        if 'amplitudes' in events.columns:
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            runinfo.amplitudes.append([amplitude] * len(event))
    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values[removeTR:lastTR+ removeTR,].T.tolist() # adding removeTR to cut the first rows
    return runinfo, str(out_motion)
#%%
subject_list = ['1005']#, '1072', '1074', '1099', '1205', '1206', '1210', '1212', '1216', '1218', '1220', '1221','1223', '1237',  '1245', '1247', '1254', '1258', '1266', '1268', '1269',   '1272', '1280', '1290', '1291','1301', '1303', '1309', '1312',  '1319', '1320', '1326', '1337', '1338', '1340', '1343', '1345', '1346', '1347','1350', '1357', '1359', '1362', '1374', '1376', '1378', '1379', '1384', '1388', '1389', '1392', '1393',  '1431', '1440', '1444', '1445', '1449', '1457', '1460'] # bad subject '1271', multiple runs - '1423', '030',
# Map field names to individual subject runs.


infosource = pe.Node(util.IdentityInterface(fields=['subject_id'
                                            ],
                                    ),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]

# SelectFiles - to grab the data (alternativ to DataGrabber)

templates = {'func': os.path.join(data_dir, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-task*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
             'mask': os.path.join(data_dir, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-task*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'),
             'regressors': os.path.join(data_dir, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-task*_desc-confounds_regressors.tsv'),
             'events': os.path.join('/media/Data/work/RCF_or', 'event_files', 'sub-{subject_id}.csv')}


selectfiles = pe.Node(nio.SelectFiles(templates,
                               base_directory=data_dir),
                   name="selectfiles")

#%%

# Extract motion parameters from regressors file
runinfo = pe.Node(util.Function(
    input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names', 'removeTR', 'lastTR', 'thr'],
    function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
    name='runinfo')


# Set the column names to be used from the confounds file
runinfo.inputs.regressors_names = ['std_dvars', 'framewise_displacement', 'scrub'] + \
                                   ['a_comp_cor_%02d' % i for i in range(6)]

runinfo.inputs.motion_columns   = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

runinfo.inputs.removeTR = removeTR
runinfo.inputs.lastTR = lastTR
runinfo.inputs.thr = thr # set threshold of scrubbing
#%%
skip = pe.Node(interface=fsl.ExtractROI(), name = 'skip') 
skip.inputs.t_min = removeTR
skip.inputs.t_size = lastTR

#%%

susan = create_susan_smooth()
susan.inputs.inputnode.fwhm = fwhm

#%%
def changeTostring(arr):
    return arr[0]

changeTosrting = pe.Node(name="changeToString",
                         interface=Function(input_names = ['arr'],
                                            output_names = ['arr'],
                                            function = changeTostring))
#%%
modelfit = pe.Workflow(name='fsl_fit', base_dir= output_dir)
"""
Use :class:`nipype.algorithms.modelgen.SpecifyModel` to generate design information.
"""

modelspec = pe.Node(interface=model.SpecifyModel(),                  
                    name="modelspec")

modelspec.inputs.input_units = 'secs'
modelspec.inputs.time_repetition = tr
modelspec.inputs.high_pass_filter_cutoff= 120
"""
Use :class:`nipype.interfaces.fsl.Level1Design` to generate a run specific fsf
file for analysis
"""

## Building contrasts
level1design = pe.Node(interface=fsl.Level1Design(), name="level1design")



# set contrasts, depend on the condition
cond_names = ['Aplus1','Bplus1','US_Bplus1','US_Aplus1', 'minus1', 'Aplus2','Bplus2','US_Bplus2','US_Aplus2', 'minus2']

cont1 = ('stim', 'T', cond_names, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
cont2 = ('P>M', 'T', cond_names, [0.125, 0.125, 0.125, 0.125, -0.5, 0.125, 0.125, 0.125, 0.125, -0.5])
cont3 = ('shock_general', 'T', cond_names, [0, 0, 0.25, 0.25, 0, 0, 0, 0.25, 0.25, 0])
cont4 = ('Shock_NoShockGeneral', 'T', cond_names, [-0.25, -0.25, 0.25, 0.25, 0, -0.25, -0.25, 0.25, 0.25, 0])
cont5 = ('CSplus2>CSplus1', 'T', cond_names, [-0.25, -0.25, -0.25, -0.25, 0, 0.25, 0.25, 0.25, 0.25, 0])
cont6 = ('CSnoShock2 > CSnoshock1', 'T', cond_names, [-0.5, -0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0])
cont7 = ('CSShock2 > CSshock1', 'T', cond_names, [0, 0, -0.5, -0.5, 0 , 0, 0, 0.5, 0.5, 0])
cont8 = ('CSPlus2 > CSminus2', 'T', cond_names, [0, 0, 0, 0, 0 , 0.25, 0.25, 0.25, 0.25, -1])

contrasts = [cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8]

level1design.inputs.interscan_interval = tr
level1design.inputs.bases = {'dgamma': {'derivs': False}}
level1design.inputs.contrasts = contrasts
level1design.inputs.model_serial_correlations = True    
"""
Use :class:`nipype.interfaces.fsl.FEATModel` to generate a run specific mat
file for use by FILMGLS
"""

modelgen = pe.MapNode(
    interface=fsl.FEATModel(),
    name='modelgen',
    iterfield=['fsf_file', 'ev_files'])
"""
Use :class:`nipype.interfaces.fsl.FILMGLS` to estimate a model specified by a
mat file and a functional run
"""
mask =  pe.Node(interface= fsl.maths.ApplyMask(), name = 'mask')


modelestimate = pe.MapNode(
    interface=fsl.FILMGLS(smooth_autocorr=True, mask_size=5, threshold=100),
    name='modelestimate',
    iterfield=['design_file', 'in_file', 'tcon_file'])


#%%
modelfit.connect([
    (infosource, selectfiles, [('subject_id', 'subject_id')]),
    (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
    (selectfiles, skip,[('func','in_file')]),
    (skip,susan,[('roi_file','inputnode.in_files')]),
    (selectfiles, susan, [('mask','inputnode.mask_file')]),
    (susan, runinfo, [('outputnode.smoothed_files', 'in_file')]),
    (susan, modelspec, [('outputnode.smoothed_files', 'functional_runs')]),
    (runinfo, modelspec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')]),
    (modelspec, level1design, [('session_info', 'session_info')]),
    (level1design, modelgen, [('fsf_files', 'fsf_file'), ('ev_files',
                                                          'ev_files')]),
    (susan, changeTosrting, [('outputnode.smoothed_files', 'arr')]),
    (changeTosrting, mask, [('arr', 'in_file')]),
    (selectfiles, mask, [('mask', 'mask_file')]),
    (mask, modelestimate, [('out_file','in_file')]),
    (modelgen, modelestimate, [('design_file', 'design_file'),('con_file', 'tcon_file'),('fcon_file','fcon_file')]),
    
])

#%% Adding data sink
# Datasink
datasink = pe.Node(nio.DataSink(base_directory=os.path.join(output_dir, 'Sink_resp')),
                                         name="datasink")


modelfit.connect([ 
        (modelestimate, datasink, [('results_dir','1stLevel.@results')])



])
#%%
modelfit.run('MultiProc', plugin_args={'n_procs': 10})
