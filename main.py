# Copyright (c) 2026 brainlife.io

# This file is a MNE python-based brainlife.io App

# Author: Guiomar Niso Galán
# Author: Carlota Juárez Alonso
# Neuroimaging Group, Cajal Neuroscience Center, CSIC

# 03/07/2026

# Set up enviroment

import json
from pathlib import Path
import subprocess
import os 
from shutil import copyfile, rmtree, copytree
import mne
import mne_bids
import logging

# PARA EL SPOURCE ANALYSIS SI QUE NECESITO USAR UN CONT CON FREESURFER

# Logger configuration

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)

# Current path

__location__ = Path(__file__).resolve().parent

# Read the parameters from Brainlife 

config_path = __location__/'config.json'
if not config_path.exists():
    raise FileNotFoundError(f"The configuration file could not be found in {config_path}")

with open (config_path, 'r') as f:
    config = json.load(f)

# Input paths 
# The output of the last app is now configurated as the input (bids_root key)
bids_root = config.get('bids_dir')
if not bids_root:
    raise ValueError("'bids_dir' parameter is required")

bids_root_path = Path(bids_root).resolve()

# Output paths

deriv_root = __location__/'out_dir'
html_report_dir = __location__/'html_report'

# Ensure output directories exist

if deriv_root.exists():
    rmtree(deriv_root)
html_report_dir.mkdir(parents = True, exist_ok = True)

# Copy the input folder ('bids_root') in the output folder ('out_dir') to have all the data there

copytree(bids_root_path, deriv_root, dirs_exist_ok = True)

# Rewrite the info in the .json file into a .py file

file_name = __location__/'pipeline_config.py'

# Inputs from the interface web to MNE variables

with open(file_name, 'w') as f:

    f.write(f"bids_root = '{bids_root_path}'\n")
    f.write(f"deriv_root = '{deriv_root}'\n")

    data_type = config.get('data_type')
    if not data_type:
        raise ValueError("'data_type' parameter is required (must be 'eeg' or 'meg')")
    f.write(f"data_type = '{data_type}'\n")

    if data_type == 'eeg':
        ch_types = ['eeg']
    else:
        meg_ch_types = config.get('meg_ch_types', 'meg')
        ch_types = [meg_ch_types]
    f.write(f"ch_types = {ch_types}\n")

    # General settings (always nedeed)

    subject = '001'
    f.write(f"subjects = ['{subject}']\n")

    task = config.get('task', None)
    if task:
        f.write(f"task = '{task}'\n")
    else:
        raise ValueError("'task' parameter is required")  

    task_is_rest = config.get('task_is_rest', False)
    f.write(f"task_is_rest = {task_is_rest}\n")

    conditions = config.get('conditions', None)
    if conditions:
        f.write(f"conditions = {conditions}\n")
    elif not task_is_rest:
        raise ValueError("'conditions' parameter is required unless task_is_rest is True")

    interactive = config.get('interactive', False)
    f.write(f"interactive = {interactive}\n")
    
    # --------------
    run_source_estimation = config.get('run_source_estimation', True)
    if task_is_rest and not conditions and run_source_estimation:
        logger.warning("task_is_rest=True and no 'conditions' were provided, no evoked data created at the sensor-analysis stage, so source estimation cannot run for this dataset")
        run_source_estimation = False
        #the following steps will not be run, the html report will be the same as the one after sensor analysis
    f.write(f"run_source_estimation = {run_source_estimation}\n")

    subjects_dir = config.get('subjects_dir', None)
    if not subjects_dir:
        subjects_dir = deriv_root/'freesurfer'/'subjects'
        subjects_dir.mkdir(parents = True, exist_ok = True)
    f.write(f"subjects_dir = '{subjects_dir}'\n")

    # BEM surface
    use_template_mri = config.get('use_template_mri', None)
    if use_template_mri:
        f.write(f"use_template_mri = '{use_template_mri}'\n")
    
    adjust_coreg = config.get('adjust_coreg', False)
    f.write(f"adjust_coreg = {adjust_coreg}\n")

    bem_mri_images = config.get('bem_mri_images', 'auto')
    if bem_mri_images:
        f.write(f"bem_mri_images = '{bem_mri_images}'\n")

    recreate_bem = config.get('recreate_bem', False)
    f.write(f"recreate_bem = {recreate_bem}\n")

    recreate_scalp_surface = config.get('recreate_scalp_surface', False)
    f.write(f"recreate_scalp_surface = {recreate_scalp_surface}\n")

    freesurfer_verbose = config.get('freesurfer_verbose', False)
    f.write(f"freesurfer_verbose = {freesurfer_verbose}\n")

    # Source space and forward solution

    mri_t1_path_generator = config.get('mri_t1_path_generator', None)
    if mri_t1_path_generator:
        f.write(f"mri_t1_path_generator = {mri_t1_path_generator}\n")

    mri_landmarks_kind = config.get('mri_landmarks_kind', None)
    if mri_landmarks_kind:
        f.write(f"mri_landmarks_kind = {mri_landmarks_kind}\n")

    spacing = config.get('spacing', 'oct6')
    if spacing:
        f.write(f"spacing = '{spacing}'\n")

    mindist = config.get('mindist', 5)
    if mindist:
        f.write(f"mindist = {mindist}\n")

    # Inverse solution

    loose = config.get('loose', 0.2)
    if loose:
        f.write(f"loose = {loose}\n")
    
    depth = config.get('depth', 0.8)
    if depth:
        f.write(f"depth = {depth}\n")

    inverse_method = config.get('inverse_method', 'dSPM')
    if inverse_method:
        f.write(f"inverse_method = '{inverse_method}'\n")
    
    noise_cov = config.get('noise_cov', (None, 0))
    if noise_cov:
        if isinstance(noise_cov, str):
            f.write(f"noise_cov = '{noise_cov}'\n")
        else:
            f.write(f"noise_cov = {noise_cov}\n")

    noise_cov_method = config.get('noise_cov_method', 'shrunk')
    if noise_cov_method:
        f.write(f"noise_cov_method = '{noise_cov_method}'\n")

    cov_rank = config.get('cov_rank', 'info')
    if cov_rank:
        if isinstance(cov_rank, str):
            f.write(f"cov_rank = '{cov_rank}'\n")
        else:
            f.write(f"cov_rank = {cov_rank}\n")

    source_info_path_update = config.get('source_info_path_update', None)
    if source_info_path_update:
        f.write(f"source_info_path_update = {source_info_path_update}\n")
    
    inverse_targets = config.get('inverse_targets', ['evoked'])
    if inverse_targets:
        f.write(f"inverse_targets = {inverse_targets}\n")

# Run python script

needs_recon_all = run_source_estimation and not use_template_mri

if needs_recon_all:
    # recon-all requires a freesurfer license file
    fs_license = config.get('fs_license', None)
    fs_home = os.environ.get('FREESURFER_HOME', '/opt/freesurfer')
    license_target = Path(fs_home)/'license.txt'
    if fs_license:
        copyfile(Path(fs_license).resolve(), license_target)
    if not license_target.exists():
        raise FileNotFoundError("Provide a valid license in the 'fs_license' parameter, or set 'use_template_mri' to skip recon-all")
    steps = "freesurfer,source"
else:
    steps = "source"

command = ["mne_bids_pipeline", f"--config={file_name}", f"--steps={steps}"]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    raise e

'''
command = ["mne_bids_pipeline", f"--config={file_name}", "--steps=source"]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    raise e
'''

# Find the reports and make a copy in out_html folder

real_deriv_root = deriv_root.resolve()

for path in real_deriv_root.rglob("*.html"):
    if "sub-average" not in path.name:
        logger.info(f"{path.name} copied to the output")
        copyfile(path, html_report_dir/path.name)