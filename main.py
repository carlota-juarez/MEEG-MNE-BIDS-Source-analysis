# This file is a MNE python-based brainlife.io App

# Author: Guiomar Niso Galán
# Author: Carlota Juárez Alonso
# Neuroimaging Group, Cajal Neuroscience Center, CSIC

# 03/07/2026

# Set up environment

import json
from pathlib import Path
import subprocess
import os 
import time 
from shutil import copyfile, rmtree, copytree, copy
import logging

# Logger configuration

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("New version: 3")

# Configure PyVista so it never attempts to open a window, the plotter is created in “off-screen” mode.
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
# Configure Matplot to use the Agg backend (without a GUI)
os.environ['MPLBACKEND'] = 'Agg'
# Disable anti-aliasing in MNE's 3D rendering 
os.environ['MNE_3D_OPTION_ANTIALIAS'] = 'false'
# VTK: Which OpenGL window implementation should be used by default
os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkOSOpenGLRenderWindow'
# 3D backend that MNE will use by default
os.environ['MNE_3D_BACKEND'] = 'pyvista'

# Set up environment
import mne
import mne_bids
from mne.viz import set_3d_backend
import vtk
import pyvista as pv
pv.OFF_SCREEN = True

#  Find out the versions 
logger.info(f"MNE version: {mne.__version__}")
logger.info(f"PyVista version: {pv.__version__}")
logger.info(f"VTK version: {vtk.vtkVersion().GetVTKVersion()}")

# 3D IMAGE GENERATION
def generate_interactive_3d_report(subjects_dir, fs_subject, deriv_root, html_report_dir, subject):
    # Returns a list of (visible_label, html_filename) only those that were successfully generated.
    # sets the MNE 3D backend right before drawing
    set_3d_backend("pyvista")
    
    # Folder named 'interactive_3d' inside the report directory
    interactive_dir = html_report_dir / 'interactive_3d'
    interactive_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialization of the list to be returned
    generated = []
    # 1. Co-registration and BEM surfaces
    try:
        # Search recursively within deriv_root for the files 
        trans_candidates = sorted(deriv_root.rglob(f"sub-{subject}*trans.fif"))
        info_candidates = sorted(deriv_root.rglob(f"sub-{subject}*raw.fif")) or sorted(deriv_root.rglob(f"sub-{subject}*epo.fif"))
        if not trans_candidates or not info_candidates:
            logger.warning("No trans.fif/raw.fif files detected")
        else:
            # If there are candidates, take the first one from info_candidates
            info = mne.io.read_info(str(info_candidates[0]))
            try:
                # Generate the co-registered 3D image
                fig = mne.viz.plot_alignment(
                    info=info,
                    trans=str(trans_candidates[0]),
                    subject=fs_subject,
                    subjects_dir=str(subjects_dir),
                    surfaces=['head-dense', 'inner_skull', 'outer_skull'],
                    coord_frame='mri',
                    show_axes=True,
                    show = False,
                )
            except Exception:
                # If there are no BEM surfaces, try with a simpler version, showing only the surface of the head 
                fig = mne.viz.plot_alignment(
                    info=info,
                    trans=str(trans_candidates[0]),
                    subject=fs_subject,
                    subjects_dir=str(subjects_dir),
                    surfaces=['head'],
                    coord_frame='mri',
                    show_axes=True,
                    show = False,
                )
            out_path = interactive_dir / f'sub-{subject}_coreg_bem.html'
            # Static screenshot of the 3D figure
            fig.plotter.screenshot(interactive_dir / "alignment.png")
            try:
                # Export the scene to interactive HTML
                fig.plotter.export_html(interactive_dir/"alignment.html")
                fig.plotter.export_html(str(out_path))
            except Exception as err:
                logger.warning(err)
            fig.plotter.close()
            # Record the readable label and the file name in the list of results 
            generated.append(('Co-registration and BEM surfaces', out_path.name))
            logger.info(f"Interactive figure saved in {out_path}")
    except Exception as e:
        logger.warning(f"The interactive figure could not be generated: {e}")
        
    # 2. Source estimate
    try:
        # Search for source estimation files in .stc format
        stc_candidates = sorted(deriv_root.rglob(f"sub-{subject}*-lh.stc"))
        if not stc_candidates:
            logger.warning("No -lh.stc file detected")
        else:
            # Take the first candidate and remove the suffix -lh.stc
            stc_stem = str(stc_candidates[0])[:-len('-lh.stc')]
            stc = mne.read_source_estimate(stc_stem)
            # Draw the 3D brain using the overlay activity
            brain = stc.plot(
                subject=fs_subject,
                subjects_dir=str(subjects_dir),
                hemi='both',
                backend='pyvista',
                time_viewer=False,
                show_traces=False,
                show=False,
            )
            out_path = interactive_dir / f'sub-{subject}_source_estimate.html'
            # Static screenshot of the 3D figure
            brain.save_image(interactive_dir/"source.png")
            try:
                # Export the scene to interactive HTML
                brain.plotter.export_html(interactive_dir/"brain.html")
                brain.plotter.export_html(str(out_path))
            except Exception as err:
                logger.warning(err)
            brain.close()
            # Record the readable label and the file name in the list of results 
            generated.append(('Source estimate', out_path.name))
            logger.info(f"Source estimate figure saved in {out_path}")
    except Exception as e:
        logger.warning(f"The interactive figure could not be generated: {e}")          

    # 3. HTML index
    if generated:
        index_path = interactive_dir / 'index.html'
        with open(index_path, 'w', encoding='utf-8') as idx:
            idx.write("<html><head><meta charset='utf-8'>"
                       "<title></title>Interactive 3D Visualizations</head><body>")
            idx.write(f"<h1>Sub-{subject}: Interactive 3D Visualizations</h1>")
            idx.write("<p>Drag with the left mouse button to rotate and scroll the mouse wheel to zoom</p>")
            for label, filename in generated:
                idx.write(f"<h2>{label}</h2>")
                idx.write(f"<iframe src='{filename}' width='100%' height='700' "
                          f"style='border:none;'></iframe>")
            idx.write("</body></html>")
 
    return generated

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
    f.write("import mne\n")
    f.write("mne.viz.set_3d_backend('pyvista')\n\n")
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
        # The following steps will not be run, the html report will be the same as the one after sensor analysis
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
        f.write(f"mri_t1_path_generator = '{mri_t1_path_generator}'\n")

    mri_landmarks_kind = config.get('mri_landmarks_kind', None)
    if mri_landmarks_kind:
        f.write(f"mri_landmarks_kind = '{mri_landmarks_kind}'\n")

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
        f.write(f"source_info_path_update = '{source_info_path_update}'\n")
    
    inverse_targets = config.get('inverse_targets', ['evoked'])
    if inverse_targets:
        f.write(f"inverse_targets = {inverse_targets}\n")

    # When running source analysis is desired (run_source_estimation = True) and use_template_montage is empty
    # The subject's actual anatomy will be used and not a standard template
    needs_recon_all = run_source_estimation and not use_template_mri

    if not run_source_estimation:
        # Nothing to run at this stage. The user disabled source estimation or it was force disable because this is resting-state data
        steps = None

    elif needs_recon_all:

        if t1_path is None or not t1_path.exists():
            raise FileNotFoundError("A T1w es needed to execute recon-all or set 'use_template_mri' to skip it")
        # Copy the t1w file to the BIDS directory 
        anat_dir = deriv_root/f'sub-{subject}'/'anat'
        anat_dir.mkdir(parents=True, exist_ok=True)
        extension = "".join(t1_path.suffixes)
        copyfile(t1_path, anat_dir/f'sub-{subject}_T1w{extension}')

        # Absolute path
        license_target = Path(os.getcwd()).resolve() / 'license.txt'
        # recon-all requires a freesurfer license file
        fs_license = config.get('fs_license', None)
        if fs_license and fs_license.strip() != "":
            with open(license_target, 'w') as file:
                file.write(fs_license.strip() + "\n")
        
        if license_target.exists():
            f.write(f"freesurfer_license = '{str(license_target.resolve())}'\n")
            # Además de escribirlo en el archivo, se lo inyectamos a la fuerza al entorno de procesos de Python
            os.environ['FS_LICENSE'] = str(license_target.resolve())
        else:
            raise FileNotFoundError("Provide a valid license in the 'fs_license' parameter or set 'use_template_mri' to skip recon-all")
        
        steps = "freesurfer,source"

    else:
        if use_template_mri:
            f.write(f"use_template_mri = '{use_template_mri}'\n")
        steps = "source"

pipeline_start_time = time.time()

# Run python script

if steps is not None:
    command = ["mne_bids_pipeline", f"--config={file_name}", f"--steps={steps}"]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise e
else:
    logger.info("run_source_estimation is False: skipping mne_bids_pipeline execution")

# CREATE INTERACTIVE 3D VISUALIZATION
# fs_subject: FreeSurfer subject that actually contains the surfaces
# (fsaverage if a template was used, or the subject reconstructed with recon-all)
if use_template_mri:
    fs_subject = use_template_mri
elif (subjects_dir/f"sub-{subject}").exists():
    fs_subject = f"sub-{subject}"
else:
    fs_subject = subject

# Display the content of deriv_root
logger.info("Contenido de deriv_root")
for p in sorted(deriv_root.rglob("*")):
    if p.is_file() and p.suffix in ('.fif', '.stc', '.gz'):
        logger.info(str(p.relative_to(deriv_root)))
logger.info("End of deriv_root content")

try:
    generated_3d_figures = generate_interactive_3d_report(
        subjects_dir=subjects_dir,
        fs_subject=fs_subject,
        deriv_root=deriv_root,
        html_report_dir=html_report_dir,
        subject=subject,
    )
except Exception as e:
    logger.warning(f"The interactive 3D visualizations could not be generated: {e}")
    generated_3d_figures = []

# Find the reports and make a copy in out_html folder

real_deriv_root = deriv_root.resolve()

interactive_link_html = (
    "<div style='padding:12px;margin-top:16px;background:#eef3ff;"
    "border-top:2px solid #6699cc;font-family:sans-serif;'>"
    "<a href='interactive_3d/index.html' target='_blank'>"
    "Drag with the left mouse button to rotate and scroll the mouse wheel to zoom</a></div>"
)

for path in real_deriv_root.rglob("*.html"):
    if "sub-average" not in path.name:
        logger.info(f"{path.name} copied to the output")
        dest = html_report_dir/path.name
        copyfile(path, dest)

        # Only reports generated by this App should get the interactive link 
        # The time this file was modified is the same as or later than the time the pipeline started?
        report_by_this_app = path.stat().st_mtime >= pipeline_start_time
        if generated_3d_figures and report_by_this_app:
            try:
                content = dest.read_text(encoding='utf-8')
                if "</body>" in content:
                    content = content.replace("</body>", interactive_link_html + "</body>")
                else:
                    content += interactive_link_html
                dest.write_text(content, encoding='utf-8')
            except Exception as e:
                logger.warning(f"The link to the interactive figures could not be inserted in {dest.name}: {e}")
