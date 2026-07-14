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

# hay que fijarlas antes de importar pyvista/mne.viz
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MNE_3D_OPTION_ANTIALIAS'] = 'false'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

# Desactivamos el backend 3D interactivo para evitar que VTK busque una ventana física
os.environ['MNE_3D_BACKEND'] = 'pyvista'

from shutil import copyfile, rmtree, copytree, copy
import mne
import mne_bids
import logging

# Logger configuration

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)

# GENERACIÓN DE FIGURAS 3D
def generate_interactive_3d_report(subjects_dir, fs_subject, deriv_root, html_report_dir, subject):
    # Genera figuras 3D interactivas. 
    # Devuelve una lista de tuplas (etiqueta, nombre_de_fichero) con lo que se generó correctamente.
    import pyvista as pv
    pv.OFF_SCREEN = True
    mne.viz.set_3d_backend('pyvista')
    
    # directorio de las figuras intercativas
    interactive_dir = html_report_dir / 'interactive_3d'
    interactive_dir.mkdir(parents=True, exist_ok=True)
    
    # inicialización de la lista
    generated = []
    # coregistro y superficies BEM
    try:
        trans_candidates = sorted(deriv_root.rglob(f"sub-{subject}*trans.fif"))
        info_candidates = sorted(deriv_root.rglob(f"sub-{subject}*raw.fif")) or sorted(deriv_root.rglob(f"sub-{subject}*epo.fif"))
        if not trans_candidates or not info_candidates:
            logger.warning("No trans.fif/raw.fif files detected")
        else:
            info = mne.io.read_info(str(info_candidates[0]))
            try:
                fig = mne.viz.plot_alignment(
                    info=info,
                    trans=str(trans_candidates[0]),
                    subject=fs_subject,
                    subjects_dir=str(subjects_dir),
                    surfaces=['head-dense', 'inner_skull', 'outer_skull'],
                    coord_frame='mri',
                    show_axes=True,
                )
            except Exception:
                # si no existen superficies BEM
                fig = mne.viz.plot_alignment(
                    info=info,
                    trans=str(trans_candidates[0]),
                    subject=fs_subject,
                    subjects_dir=str(subjects_dir),
                    surfaces=['head'],
                    coord_frame='mri',
                    show_axes=True,
                )
            out_path = interactive_dir / f'sub-{subject}_coreg_bem.html'
            fig.plotter.export_html(str(out_path))
            fig.plotter.close()
            generated.append(('Coregistracion y superficies BEM', out_path.name))
            logger.info(f"Figura interactiva de coregistro/BEM guardada en {out_path}")
    except Exception as e:
        logger.warning(f"No se pudo generar la figura interactiva de coregistro/BEM: {e}")
    # ESTIMACION DE FUENTES 
    try:
        stc_candidates = sorted(deriv_root.rglob(f"sub-{subject}*-lh.stc"))
        if not stc_candidates:
            logger.warning("No se encontro ningun fichero -lh.stc; se omite la figura de fuente")
        else:
            stc_stem = str(stc_candidates[0])[:-len('-lh.stc')]
            stc = mne.read_source_estimate(stc_stem)
            brain = stc.plot(
                subject=fs_subject,
                subjects_dir=str(subjects_dir),
                hemi='both',
                backend='pyvista',
                time_viewer=False,
                show_traces=False,
            )
            out_path = interactive_dir / f'sub-{subject}_source_estimate.html'
            brain.plotter.export_html(str(out_path))
            brain.close()
            generated.append(('Estimacion de fuente', out_path.name))
            logger.info(f"Figura interactiva de la estimacion de fuente guardada en {out_path}")
    except Exception as e:
        logger.warning(f"No se pudo generar la figura interactiva de la estimacion de fuente: {e}")          

    # indice para todas las figuras
    if generated:
        index_path = interactive_dir / 'index.html'
        with open(index_path, 'w', encoding='utf-8') as idx:
            idx.write("<html><head><meta charset='utf-8'>"
                       "<title>Visualizaciones 3D interactivas</title></head><body>")
            idx.write(f"<h1>Sub-{subject}: visualizaciones 3D interactivas</h1>")
            idx.write("<p>Arrastra con el boton izquierdo para rotar, "
                       "rueda del raton para zoom.</p>")
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
    
    f.write("import os\n")
    f.write("os.environ['PYVISTA_OFF_SCREEN'] = 'true'\n")
    f.write("os.environ['MPLBACKEND'] = 'Agg'\n")
    f.write("os.environ['MNE_3D_OPTION_ANTIALIAS'] = 'false'\n\n")

    f.write("os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n")
    f.write("os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')\n\n")

    #Con vtk-osmesa el renderizado es 100% por software
    f.write("import pyvista\n")
    f.write("pyvista.OFF_SCREEN = True\n")

    f.write("import mne\n")
    f.write("mne.viz.set_3d_backend('pyvista')\n\n")

    # -----
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

        # Ruta absoluta
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

# Run python script
command = ["mne_bids_pipeline", f"--config={file_name}", f"--steps={steps}"]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    raise e

# GENERAMOS VISUALIZACION 3D INTERACTIVAS
# fs_subject: sujeto de FreeSurfer que realmente contiene las superficies
# (fsaverage si se uso plantilla, o el sujeto reconstruido con recon-all)
if use_template_mri:
    fs_subject = use_template_mri
elif (subjects_dir/f"sub-{subject}").exists():
    fs_subject = f"sub-{subject}"
else:
    fs_subject = subject

# mostrar contenido de lo que hay en deriv_root
logger.info("Contenido de deriv_root")
for p in sorted(deriv_root.rglob("*")):
    if p.is_file() and p.suffix in ('.fif', '.stc', '.gz'):
        logger.info(str(p.relative_to(deriv_root)))
logger.info("Fin del contenido deriv_root")

try:
    generated_3d_figures = generate_interactive_3d_report(
        subjects_dir=subjects_dir,
        fs_subject=fs_subject,
        deriv_root=deriv_root,
        html_report_dir=html_report_dir,
        subject=subject,
    )
except Exception as e:
    logger.warning(f"No se pudieron generar las visualizaciones 3D interactivas: {e}")
    generated_3d_figures = []

# Find the reports and make a copy in out_html folder

real_deriv_root = deriv_root.resolve()

interactive_link_html = (
    "<div style='padding:12px;margin-top:16px;background:#eef3ff;"
    "border-top:2px solid #6699cc;font-family:sans-serif;'>"
    "<a href='interactive_3d/index.html' target='_blank'>"
    "Ver visualizaciones 3D interactivas (rota y haz zoom con el raton)</a></div>"
)

for path in real_deriv_root.rglob("*.html"):
    if "sub-average" not in path.name:
        logger.info(f"{path.name} copied to the output")
        dest = html_report_dir/path.name
        copyfile(path, dest)
        if generated_3d_figures:
            try:
                content = dest.read_text(encoding='utf-8')
                if "</body>" in content:
                    content = content.replace("</body>", interactive_link_html + "</body>")
                else:
                    content += interactive_link_html
                dest.write_text(content, encoding='utf-8')
            except Exception as e:
                logger.warning(f"No se pudo insertar el enlace a las figuras interactivas en {dest.name}: {e}")
