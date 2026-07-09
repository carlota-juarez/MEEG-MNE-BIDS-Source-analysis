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
from shutil import copyfile, rmtree, copytree, copy
import mne
import mne_bids
import logging

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

t1 = config.get('t1', None)
if t1 == "" or t1 == "null":
    t1_path = None

if t1:
    t1_path = Path(t1).resolve()
else:
    t1_path = None

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
    f.write(f"t1 = '{t1}'\n")

    data_type = config.get('data_type')
    if not data_type:
        raise ValueError("'data_type' parameter is required (must be 'eeg' or 'meg')")
    f.write(f"data_type = '{data_type}'\n")

    if data_type == 'eeg':
        ch_types = ['eeg']
        eeg_template_montage = config.get('eeg_template_montage', None)
        if eeg_template_montage:
            f.write(f"eeg_template_montage = '{eeg_template_montage}'\n")
    else:
        meg_ch_types = config.get('meg_ch_types', 'meg')
        ch_types = [meg_ch_types]
    f.write(f"ch_types = {ch_types}\n")

    # General settings (always nedeed)

    subject = '01'
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
    subjects_dir = Path(subjects_dir)
    subjects_dir.mkdir(parents = True, exist_ok = True)

    # Avoid network latency by copying fsaverage from the Docker image
    fsaverage_image = Path('/opt/mne_data/subjects/fsaverage')
    target_fsaverage = subjects_dir/'fsaverage'
    if fsaverage_image.exists() and not target_fsaverage.exists():
        copytree(baked_fsaverage, target_fsaverage)

    f.write(f"subjects_dir = '{subjects_dir}'\n")

    # BEM surface
    # When this parameter is not defined, FreeSurfer runs recon-all
    use_template_mri = config.get('use_template_mri', None)
    if use_template_mri == "" or use_template_mri == "null":
        use_template_mri = None
    
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
    if spacing is not None:
        f.write(f"spacing = '{spacing}'\n")

    mindist = config.get('mindist', 5)
    if mindist is not None:
        f.write(f"mindist = {mindist}\n")

    # Inverse solution

    loose = config.get('loose', 0.2)
    if loose is not None:
        f.write(f"loose = {loose}\n")
    
    depth = config.get('depth', 0.8)
    if depth is not None:
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

    # When running source analysis is desired (run_source_estimation = True) and use_template_montage is empty
    # The subject's actual anatomy will be used and not a standard template
    needs_recon_all = run_source_estimation and not use_template_mri

    if needs_recon_all:

        if t1_path is None or not t1_path.exists():
            raise FileNotFoundError("A T1w es needed to execute recon-all or set 'use_template_mri' to skip it")
        # Copy the t1w file to the BIDS directory 
        anat_dir = deriv_root/f'sub-{subject}'/'anat'
        anat_dir.mkdir(parents=True, exist_ok=True)
        extension = "".join(t1_path.suffixes)
        copyfile(t1_path, anat_dir/f'sub-{subject}_T1w{extension}')

        # recon-all requires a freesurfer license file
        fs_license = config.get('fs_license', None)
        license_target = __location__/'license.txt'
        if fs_license:
            with open(license_target, 'w') as file:
                file.write(fs_license.strip() + "\n")
        
        if license_target.exists():
            f.write(f"freesurfer_license = '{str(license_target.resolve())}'\n")
        else:
            raise FileNotFoundError("Provide a valid license in the 'fs_license' parameter or set 'use_template_mri' to skip recon-all")
        
        steps = "freesurfer,source"

    else:
        if use_template_mri:
            f.write(f"use_template_mri = '{use_template_mri}'\n")
        steps = "source"

# Run python script
command = ["mne_bids_pipeline", f"--config={file_name}", f"--steps={steps}"]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    raise e

# Find the reports and make a copy in out_html folder

real_deriv_root = deriv_root.resolve()

for path in real_deriv_root.rglob("*.html"):
    if "sub-average" not in path.name:
        logger.info(f"{path.name} copied to the output")
        copyfile(path, html_report_dir/path.name)