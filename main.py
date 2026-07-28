# This file is a MNE python-based brainlife.io App

# Author: Carlota Juárez Alonso
# Author: Guiomar Niso Galán
# Neuroimaging Group, Cajal Neuroscience Center, CSIC

# 03/07/2026

# Set up environment

import json
from pathlib import Path
import subprocess
import os 
from shutil import copyfile, rmtree, copytree 
import logging
import numpy as np
import re

# Logger configuration

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure PyVista so it never attempts to open a window, the plotter is created in “off-screen” mode.
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
# Configure Matplot to use the Agg backend (without a GUI)
os.environ['MPLBACKEND'] = 'Agg'
# Disable anti-aliasing in MNE's 3D rendering 
os.environ['MNE_3D_OPTION_ANTIALIAS'] = 'false'
# VTK: Which OpenGL window implementation should be used by default
os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkOSOpenGLRenderWindow'
# 3D backend that MNE will use by default
os.environ['MNE_3D_BACKEND'] = 'pyvistaqt'

# Set up environment
import mne
from mne.coreg import get_mni_fiducials
from mne.channels import make_dig_montage
from mne.transforms import Transform
from mne.io.constants import FIFF
from mne_bids import BIDSPath, get_anat_landmarks, write_anat
import pyvista as pv
pv.OFF_SCREEN = True

if not os.environ.get('DISPLAY'):
    pv.start_xvfb(wait=3)

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
if t1 and t1 not in ("", "null"):
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
    #----------------------------------------------------------------------------------------------
    # BrainLife runs in headless mode, so we need configuration variables to ensure that the figures are generated in the background
    # Write instructions in the configuration file to set up 3D rendering before the pipeline runs
    f.write("import os\n")
    f.write("os.environ['PYVISTA_OFF_SCREEN'] = 'true'\n")
    f.write("os.environ['MPLBACKEND'] = 'Agg'\n")
    f.write("os.environ['MNE_3D_OPTION_ANTIALIAS'] = 'false'\n\n")
    f.write("os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkOSOpenGLRenderWindow'\n\n")
    f.write("import pyvista\n")
    f.write("pyvista.OFF_SCREEN = True\n")
    f.write("import nest_asyncio\n")
    f.write("nest_asyncio.apply()\n")
    f.write("import mne\n")
    f.write("mne.viz.set_3d_backend('pyvistaqt')\n\n")
    # ---------------------------------------------------------------------------------------------

    f.write(f"bids_root = '{bids_root_path}'\n")
    f.write(f"deriv_root = '{deriv_root}'\n")

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

    # General settings (always needed)

    subject = '01'
    f.write(f"subjects = ['{subject}']\n")

    # Which file to use 
    proc_priority = ['clean', 'sss', 'filt']  
    found_procs = set()
    for fp in deriv_root.rglob(f"sub-{subject}_*_raw.fif"):
        m = re.search(r"_proc-([A-Za-z0-9]+)_raw\.fif$", fp.name)
        if m:
            found_procs.add(m.group(1))

    proc_tag = next((p for p in proc_priority if p in found_procs), None)
    if proc_tag:
        f.write(f"proc = '{proc_tag}'\n")
        logger.info(f"Multiple processed versions found {sorted(found_procs)}; using proc-{proc_tag} as pipeline input")

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
    
    run_source_estimation = config.get('run_source_estimation', True)
    if task_is_rest and not conditions and run_source_estimation:
        logger.warning("task_is_rest=True and no 'conditions' were provided, no evoked data created at the sensor-analysis stage, so source estimation cannot run for this dataset")
        run_source_estimation = False
    f.write(f"run_source_estimation = {run_source_estimation}\n")

    subjects_dir = config.get('subjects_dir', None)
    if not subjects_dir:
        subjects_dir = deriv_root/'freesurfer'/'subjects'
    subjects_dir = Path(subjects_dir)
    subjects_dir.mkdir(parents = True, exist_ok = True)

    # Avoid network latency by copying fsaverage from the Docker image
    fsaverage_image = Path('/opt/freesurfer/subjects/fsaverage')
    target_fsaverage = subjects_dir/'fsaverage'
    if fsaverage_image.exists() and not target_fsaverage.exists():
        copytree(fsaverage_image, target_fsaverage)

    f.write(f"subjects_dir = r'{subjects_dir}'\n")

    use_template_mri = config.get('use_template_mri', None)
    if use_template_mri in ("", "null"):
        use_template_mri = None

    needs_recon_all = run_source_estimation and not use_template_mri
    if use_template_mri:
        f.write(f"use_template_mri = '{use_template_mri}'\n")

    if needs_recon_all:
        # Limit parallel jobs to reduce OOM risk on Brainlife compute nodes
        f.write("n_jobs = 2\n")

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
    # Remove? -------------
    mri_t1_path_generator = config.get('mri_t1_path_generator', None)
    if mri_t1_path_generator:
        f.write(f"mri_t1_path_generator = '{mri_t1_path_generator}'\n")

    mri_landmarks_kind = config.get('mri_landmarks_kind', None)
    if mri_landmarks_kind:
        f.write(f"mri_landmarks_kind = '{mri_landmarks_kind}'\n")
    # ---------
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

    if not (isinstance(noise_cov, str) and noise_cov == 'emptyroom'):
        f.write("process_empty_room = False\n")
    if not (isinstance(noise_cov, str) and noise_cov == 'rest'):
        f.write("process_rest = False\n")

    noise_cov_method = config.get('noise_cov_method', 'shrunk')
    if noise_cov_method:
        f.write(f"noise_cov_method = '{noise_cov_method}'\n")

    cov_rank = config.get('cov_rank', 'info')
    if cov_rank:
        if isinstance(cov_rank, str):
            f.write(f"cov_rank = '{cov_rank}'\n")
        else:
            f.write(f"cov_rank = {cov_rank}\n")
    # remove?
    source_info_path_update = config.get('source_info_path_update', None)
    if source_info_path_update:
        f.write(f"source_info_path_update = '{source_info_path_update}'\n")
    # -----------------------
    
    inverse_targets = config.get('inverse_targets', ['evoked'])
    if inverse_targets:
        f.write(f"inverse_targets = {inverse_targets}\n")


def write_t1w_coreg_landmarks(subject, subjects_dir, bids_root_path, t1w_bids_file):
    fs_subject = f"sub-{subject}"

    # Fiducials estimated in the FreeSurfer "mri" 
    fids_mri = get_mni_fiducials(fs_subject, subjects_dir=subjects_dir)
    fid_by_ident = {p['ident']: p['r'] for p in fids_mri}

    dig_montage = make_dig_montage(
        nasion=fid_by_ident[FIFF.FIFFV_POINT_NASION],
        lpa=fid_by_ident[FIFF.FIFFV_POINT_LPA],
        rpa=fid_by_ident[FIFF.FIFFV_POINT_RPA],
        coord_frame='head',
    )
    info = mne.create_info(ch_names=['fiducial_placeholder'], sfreq=1000.0, ch_types='misc')
    info.set_montage(dig_montage, on_missing='ignore')

    identity_trans = Transform('head', 'mri', np.eye(4))

    landmarks = get_anat_landmarks(
        image=t1w_bids_file,
        info=info,
        trans=identity_trans,
        fs_subject=fs_subject,
        fs_subjects_dir=subjects_dir,
    )

    t1w_bids_path = BIDSPath(
        subject=subject, root=bids_root_path, datatype='anat', suffix='T1w',
    )
    write_anat(image=t1w_bids_file, bids_path=t1w_bids_path, landmarks=landmarks, overwrite=True)
    logger.info(f"Wrote estimated NAS/LPA/RPA landmarks to the T1w JSON sidecar for sub-{subject}")

# Determine pipeline steps
if not run_source_estimation:
    steps = None
elif needs_recon_all:
    steps = "freesurfer,source"
else:
    steps = "source"

# FreeSurfer recon-all setup (subject anatomy, not template)
if needs_recon_all:
    if t1_path is None or not t1_path.exists():
        raise FileNotFoundError(
            "A T1w MRI is required to run recon-all. "
            "Provide it via the 't1' parameter or set 'use_template_mri' to 'fsaverage' to skip recon-all."
        )

    extension = "".join(t1_path.suffixes)
    for target_root in (bids_root_path, deriv_root):
        anat_dir = target_root / f'sub-{subject}' / 'anat'
        anat_dir.mkdir(parents=True, exist_ok=True)
        copyfile(t1_path, anat_dir / f'sub-{subject}_T1w{extension}')

    original_fs_home = Path(os.environ.get('FREESURFER_HOME', '/opt/freesurfer'))
    license_target = __location__ / 'freesurfer_license.txt'

    fs_license = config.get('fs_license', None)
    if fs_license and fs_license.strip() != "":
        with open(license_target, 'w') as file:
            file.write(fs_license.strip() + "\n")
        logger.info("Using FreeSurfer license provided by the user via 'fs_license' parameter")
    elif not license_target.exists():
        # 1) Standard case: the compute resource exports FS_LICENSE pointing to a file
        resource_license = os.environ.get('FS_LICENSE')
        candidate_paths = []
        if resource_license:
            candidate_paths.append(Path(resource_license))
            logger.info(f"Using FreeSurfer license from computing resource ({resource_license})")
        # 2) Fallback: some compute resources bake the license directly into the FreeSurfer install
        candidate_paths += [
            Path(os.environ.get('FREESURFER_HOME', '/opt/freesurfer')) / 'license.txt',
            Path(os.environ.get('FREESURFER_HOME', '/opt/freesurfer')) / '.license',
            Path('/opt/freesurfer/license.txt'),
            Path('/usr/local/freesurfer/license.txt'),
        ]

        for candidate in candidate_paths:
            if candidate and candidate.exists():
                copyfile(candidate, license_target)
                logger.info(f"Using FreeSurfer license from computing resource ({candidate})")
                break

    if not license_target.exists():
        raise FileNotFoundError(
            "No FreeSurfer license available. Provide one in 'fs_license' "
            "or ensure the computing resource exposes FS_LICENSE."
        )

    mni_startup = original_fs_home / 'mni' / 'share' / 'perl5' / 'MNI' / 'Startup.pm'
    if not mni_startup.exists():
        raise FileNotFoundError(
            f"FreeSurfer MNI Perl modules not found at {mni_startup}. "
            "Rebuild the Docker image with a complete FreeSurfer installation."
        )

    fs_home_mirror = __location__ / 'freesurfer_home'
    if not fs_home_mirror.exists():
        fs_home_mirror.mkdir(parents=True, exist_ok=True)
        for item in original_fs_home.iterdir():
            if item.name in ('license.txt', '.license'):
                continue
            (fs_home_mirror / item.name).symlink_to(item)
    copyfile(license_target, fs_home_mirror / 'license.txt')
 
    fs_path = str(fs_home_mirror.resolve())
    subjects_dir_str = str(subjects_dir.resolve())
    license_path = str(license_target.resolve())

    mni_dir = f"{fs_path}/mni"
    minc_bin_dir = f"{mni_dir}/bin"
    fsfast_home = f"{fs_path}/fsfast"

    os.environ['FREESURFER_HOME'] = fs_path
    os.environ['FS_LICENSE'] = license_path
    os.environ['SUBJECTS_DIR'] = subjects_dir_str
    os.environ['PERL5LIB'] = f"{fs_path}/mni/share/perl5"
    os.environ['MNI_PERL5LIB'] = f"{fs_path}/mni/share/perl5"
    os.environ['MNI_DIR'] = mni_dir
    os.environ['MINC_BIN_DIR'] = minc_bin_dir
    os.environ['FSFAST_HOME'] = fsfast_home
    os.environ['PATH'] = (
        f"{fs_path}/bin:{fs_path}/tktools:{minc_bin_dir}:{fs_path}/mni/bin:"
        + os.environ.get('PATH', '')
    )

    with open(file_name, 'a') as f:
        f.write("\n# FreeSurfer environment for recon-all\n")
        f.write(f"os.environ['FREESURFER_HOME'] = r'{fs_path}'\n")
        f.write(f"os.environ['FS_LICENSE'] = r'{license_path}'\n")
        f.write(f"os.environ['SUBJECTS_DIR'] = r'{subjects_dir_str}'\n")
        f.write(f"os.environ['PERL5LIB'] = r'{fs_path}/mni/share/perl5'\n")
        f.write(f"os.environ['MNI_PERL5LIB'] = r'{fs_path}/mni/share/perl5'\n")
        f.write(f"os.environ['MNI_DIR'] = r'{mni_dir}'\n")
        f.write(f"os.environ['MINC_BIN_DIR'] = r'{minc_bin_dir}'\n")
        f.write(f"os.environ['FSFAST_HOME'] = r'{fsfast_home}'\n")

    logger.info(f"FreeSurfer ready: FREESURFER_HOME={fs_path}, SUBJECTS_DIR={subjects_dir_str}")

# Run python script

if steps == "freesurfer,source":
    logger.info("Running the 'freesurfer' step first (recon-all)")
    fs_command = ["mne_bids_pipeline", f"--config={file_name}", "--steps=freesurfer"]
    try:
        subprocess.run(fs_command, check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        raise e

    t1w_bids_file = bids_root_path / f'sub-{subject}' / 'anat' / f'sub-{subject}_T1w{extension}'
    write_t1w_coreg_landmarks(subject, subjects_dir_str, bids_root_path, t1w_bids_file)

    logger.info("Running the 'source' step")
    source_command = ["mne_bids_pipeline", f"--config={file_name}", "--steps=source"]
    try:
        subprocess.run(source_command, check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        raise e
elif steps is not None:
    command = ["mne_bids_pipeline", f"--config={file_name}", f"--steps={steps}"]
    try:
        subprocess.run(command, check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        raise e
else:
    logger.info("run_source_estimation is False: skipping mne_bids_pipeline execution")

# Find the reports and make a copy in out_html folder

real_deriv_root = deriv_root.resolve()

for path in real_deriv_root.rglob("*.html"):
    if "sub-average" not in path.name:
        logger.info(f"{path.name} copied to the output") 
        dest = html_report_dir/path.name
        copyfile(path, dest)

