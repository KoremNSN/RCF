# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import os

from nipype import Node, Workflow #, MapNode

import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as model  # model specification

import nipype.interfaces.utility as util  # utility
import nipype.interfaces.io as nio  # Data i/o
from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces import spm
from nipype.interfaces import fsl

#%%
def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0, removeTR=4, lastTR=496, thr=0.5):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch
    from scrubFunc import scrub
    # Process the events file
    events = pd.read_csv(events_file)
    bunch_fields = ['onsets', 'durations', 'amplitudes']
    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]
    out_motion = Path('motion.par').resolve()
    #regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    regress_data = scrub(regressors_file, thr) # grab also per which will be saved as file
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


def saveScrub(regressors_file, thr):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scrubFunc import scrub
    # this function will call scrub and save a file with precentage of scrubbed framewise_displacement
    perFile = Path('percentScrub.txt').resolve()
    regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    regress_data  = scrub(regressors_file, thr) # grab also per which will be saved as file
    x = regress_data.scrub
    per = np.array([sum(x)/len(x)])
    np.savetxt(perFile, per, '%g')
    return str(perFile)

#%%
MatlabCommand.set_default_paths('/home/nachshon/Documents/MATLAB/spm12/') # set default SPM12 path in my computer. 
#fsl.FSLCommand.set_default_output_type('NIFTI_GZ')


base_root = '/media/Data/Lab_Projects/RCF'
data_root = '/media/Data/Lab_Projects/RCF/derivatives/fmriprep'
out_root = '/media/Data/Lab_Projects/RCF/work'

data_dir = data_root
output_dir = os.path.join(out_root, 'imaging')
work_dir = os.path.join(out_root, 'work') # intermediate products


subject_list1 = ['1220']
subject_list = ['020',	'029',	'030',	'038',	'1005',	'1072',	'1074',	'1099',	'1205',	'1206',	'1210',
		'1212',	'1216',	'1218',	'1220',	'1221',	'1223',	'1237',	'1245',	'1247',	'1254',	'1258',
		'1266',	'1268',	'1269',	'1271',	'1272',	'1280',	'1290',	'1291',	'1301',	'1303',	'1309',
		'1312',	'1319',	'1320',	'1326',	'1337',	'1338',	'1340',	'1343',	'1345',	'1346',	'1347',
		'1350',	'1357',	'1359',	'1362',	'1373',	'1374',	'1376',	'1378',	'1384',	'1389',	'1392',
		'1393',	'1423',	'1440',	'1444',	'1445',	'1449',	'1457',	'1460', '1388']


fwhm = 6 # smotthing paramater
tr = 1 # in seconds
removeTR = 9
lastTR = 496
thr = 0.5 # set FD scrubbing threshold

infosource = pe.Node(util.IdentityInterface(fields=['subject_id'],),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]


templates = {'func': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-task5*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
             'mask': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-task5*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz'),
             'regressors': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-task5*_desc-confounds_regressors.tsv'),
             'events': os.path.join(base_root, 'event_files', 'sub-{subject_id}.csv')}

# Flexibly collect data from disk to feed into flows.
selectfiles = pe.Node(nio.SelectFiles(templates,
                      base_directory=data_root),
                      name="selectfiles")

# selectfiles.inputs.task_id = ['556']  # task update after fixing the BDFconvert procedure

# Extract motion parameters from regressors file
runinfo = pe.MapNode(util.Function(
    input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names', 'motion_columns', 'removeTR','lastTR', 'thr'],
    function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
    name='runinfo',
    iterfield = ['in_file', 'events_file', 'regressors_file'])

runinfo.inputs.removeTR = removeTR
runinfo.inputs.lastTR = lastTR
runinfo.inputs.thr
# Set the column names to be used from the confounds file

runinfo.inputs.regressors_names = ['std_dvars', 'framewise_displacement'] + \
                                   ['a_comp_cor_%02d' % i for i in range(6)]

runinfo.inputs.motion_columns   = ['trans_x', 'trans_y','trans_z', 'rot_x', 'rot_y', 'rot_z']
 # 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2'] + \
#                                   [ 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2'] + \
#                                   ['trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2'] + \
#                                   ['rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2'] + \
#                                   ['rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2'] + \
#                                   ['rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']



svScrub = pe.Node(util.Function(
    input_names = ['regressors_file', 'thr'], output_names = ['perFile'],
    function = saveScrub), name = 'svScrub'
    )

svScrub.inputs.thr = thr
#%%
extract = pe.MapNode(fsl.ExtractROI(), name="extract", iterfield = ['in_file'])
extract.inputs.t_min = removeTR
extract.inputs.t_size = lastTR # set length of sacn
extract.inputs.output_type='NIFTI'

# smoothing
smooth = Node(spm.Smooth(), name="smooth", fwhm = fwhm)

# set contrasts, depend on the condition
cond_names = ['Aplus1', 'Aplus2', 'Bplus1', 'Bplus2', 'minus1', 'minus2', 'US_Aplus1', 'US_Aplus2', 'US_Bplus1', 'US_Bplus2']

cont1 = ('P>M',		'T', cond_names, [0.25, 0.25, 	0.25, 	0.25, 	-0.5, 	-0.5, 	0, 0, 0, 0])
cont2 = ('P2>M2', 	'T', cond_names, [0, 	0.5, 	0, 	0.5, 	0, 	-1, 	0, 0, 0, 0])
cont3 = ('P1>P2', 	'T', cond_names, [0.5, 	-0.5, 	0.5, 	-0.5, 	0, 	0, 	0, 0, 0, 0])
cont4 = ('M1>M2', 	'T', cond_names, [0, 	0, 	0, 	0, 	1, 	-1, 	0, 0, 0, 0])
cont5 = ('P1>M1', 	'T', cond_names, [0.5, 	0, 	0.5, 	0, 	-1, 	0, 	0, 0, 0, 0])
cont6 = ('M1>M2', 	'T', cond_names, [0, 	0, 	0, 	0, 	1, 	-1, 	0, 0, 0, 0])
cont7 = ('stim', 	'T', cond_names, [0.1, 	0.1, 	0.1, 	0.1, 	0.1, 	0.1, 	0.1, 0.1, 0.1, 0.1])

cont8 = ('P+S>M', 'T', cond_names, [0.125, 0.125, 0.125, 0.125, -0.5, -0.5, 0.125, 0.125, 0.125, 0.125])
cont9 = ('shock_general', 'T', cond_names, [0.25, 0.25, 0.25, 0.25, 0, 0, -0.25, -0.25, -0.25, -0.25])

contrasts = [cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9]

#%%


modelspec = Node(interface=model.SpecifySPMModel(), name="modelspec")
modelspec.inputs.concatenate_runs = False
modelspec.inputs.input_units = 'secs' # supposedly it means tr
modelspec.inputs.output_units = 'secs'
modelspec.inputs.time_repetition = 1.  # make sure its with a dot
modelspec.inputs.high_pass_filter_cutoff = 128.

level1design = pe.Node(interface=spm.Level1Design(), name="level1design") #, base_dir = '/media/Data/work')
level1design.inputs.timing_units = modelspec.inputs.output_units
level1design.inputs.interscan_interval = 1.
level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
level1design.inputs.model_serial_correlations = 'AR(1)'

# create workflow
wfSPM = Workflow(name="l1spm_resp", base_dir=work_dir)
wfSPM.connect([
        (infosource, selectfiles, [('subject_id', 'subject_id')]),
        (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
        (selectfiles, svScrub, [('regressors', 'regressors_file')]),
        (selectfiles, extract, [('func','in_file')]),
        (extract, smooth, [('roi_file','in_files')]),
        (smooth, runinfo, [('smoothed_files','in_file')]),
        (smooth, modelspec, [('smoothed_files', 'functional_runs')]),
        (runinfo, modelspec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')]),
        ])
wfSPM.connect([(modelspec, level1design, [("session_info", "session_info")])])

#%%
level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
level1estimate.inputs.estimation_method = {'Classical': 1}

contrastestimate = pe.Node(
    interface=spm.EstimateContrast(), name="contrastestimate")
contrastestimate.inputs.contrasts = contrasts


wfSPM.connect([
         (level1design, level1estimate, [('spm_mat_file','spm_mat_file')]),
         (level1estimate, contrastestimate,
            [('spm_mat_file', 'spm_mat_file'), ('beta_images', 'beta_images'),
            ('residual_image', 'residual_image')]),
    ])

#%% Adding data sink
# Datasink
datasink = Node(nio.DataSink(base_directory=os.path.join(output_dir, 'Sink_resp')),
                                         name="datasink")


wfSPM.connect([
        (level1estimate, datasink, [('beta_images',  '1stLevel.@betas.@beta_images'),
                                    ('residual_image', '1stLevel.@betas.@residual_image'),
                                    ('residual_images', '1stLevel.@betas.@residual_images'),
                                    ('SDerror', '1stLevel.@betas.@SDerror'),
                                    ('SDbetas', '1stLevel.@betas.@SDbetas'),
                ])
        ])


wfSPM.connect([
       # here we take only the contrast ad spm.mat files of each subject and put it in different folder. It is more convenient like that.
       (contrastestimate, datasink, [('spm_mat_file', '1stLevel.@spm_mat'),
                                              ('spmT_images', '1stLevel.@T'),
                                              ('con_images', '1stLevel.@con'),
                                              ('spmF_images', '1stLevel.@F'),
                                              ('ess_images', '1stLevel.@ess'),
                                              ])
        ])


#%% run
wfSPM.run('MultiProc', plugin_args={'n_procs': 14})
# wfSPM.run('Linear', plugin_args={'n_procs': 1})
