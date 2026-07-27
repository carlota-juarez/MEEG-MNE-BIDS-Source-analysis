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
from shutil import copyfile, rmtree, copytree 
import logging
import numpy as np
import re

# Logger configuration

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("New version: 3 correct")

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
from mne.viz import set_3d_backend
import vtk
import pyvista as pv
pv.OFF_SCREEN = True

if not os.environ.get('DISPLAY'):
    pv.start_xvfb(wait=3)

#  Find out the versions 
logger.info(f"MNE version: {mne.__version__}")
logger.info(f"PyVista version: {pv.__version__}")
logger.info(f"VTK version: {vtk.vtkVersion().GetVTKVersion()}")

# 3D IMAGE GENERATION
def generate_interactive_3d_report(subjects_dir, fs_subject, deriv_root, html_report_dir, subject):
    # Returns a list of (visible_label, html_filename) only those that were successfully generated.
    # sets the MNE 3D backend right before drawing
    set_3d_backend("pyvistaqt")
    
    # Folder named 'interactive_3d' inside the report directory
    interactive_dir = html_report_dir / 'interactive_3d'
    interactive_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialization of the list to be returned
    generated = []

    # STATIC FIGURES: CO-REGISTRATION, BEM AND SOURCES
    
    # 1. Co-registration and BEM surfaces
    try:
        # Search recursively within deriv_root for the files 
        trans_candidates = sorted(deriv_root.rglob(f"sub-{subject}*trans.fif"))
        info_candidates = (
            sorted(deriv_root.rglob(f"sub-{subject}*_proc-clean_raw.fif")) or 
            sorted(deriv_root.rglob(f"sub-{subject}*raw.fif")) or 
            sorted(deriv_root.rglob(f"sub-{subject}*epo.fif")) or
            sorted(deriv_root.rglob(f"sub-{subject}*ave.fif"))
        )
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
                )
            png_path = interactive_dir / f'sub-{subject}_coreg_bem.png'
            # Static screenshot of the 3D figure
            fig.plotter.screenshot(str(png_path))
            fig.plotter.close()
            # Record the readable label and the file name in the list of results 
            generated.append(('Co-registration and BEM surfaces (Static)', png_path.name, 'image'))
            logger.info(f"Static figure saved in {png_path}")
    except Exception as e:
        logger.warning(f"The static figure could not be generated: {e}")
        
    # 2. Source estimate
    try:
        # Search for source estimation files in .stc format
        stc_candidates = sorted(deriv_root.rglob(f"sub-{subject}*+hemi.h5")) or \
                         sorted(deriv_root.rglob(f"sub-{subject}*-lh.stc"))
        if stc_candidates:
            stc_file = str(stc_candidates[0])
            if stc_file.endswith('-lh.stc'):
                stc_file = stc_file[:-len('-lh.stc')]
            stc = mne.read_source_estimate(stc_file)
            # Draw the 3D brain using the overlay activity
            brain = stc.plot(
                subject=fs_subject,
                subjects_dir=str(subjects_dir),
                hemi='both',
                backend='pyvistaqt',
                time_viewer=False,
                show_traces=False,
            )
            png_path = interactive_dir / f'sub-{subject}_source_estimate.png'
            # Static screenshot of the 3D figure
            brain.save_image(str(png_path))
            brain.close()
            # Record the readable label and the file name in the list of results 
            generated.append(('Source estimate (Static)', png_path.name, 'image'))
            logger.info(f"Static source estimate figure saved in {png_path}")
    except Exception as e:
        logger.warning(f"The interactive figure could not be generated: {e}")          

    # 3D INTERACTIVE MODEL (static anatomy)

    # Create a 3D view of the cerebral cortex using MNE and Three.js
    try:
        subj_path = Path(subjects_dir) / fs_subject / 'surf'
        lh_pial = subj_path / 'lh.pial'
        rh_pial = subj_path / 'rh.pial'
        
        vertices_list = []
        faces_list = []
        vertex_offset = 0

        # Read the grids for both hemispheres
        for surf_path in [lh_pial, rh_pial]:
            if surf_path.exists():
                coords, faces = mne.read_surface(str(surf_path))
                logger.info(f"Surface {surf_path.name} -> Vertex: {len(coords)}, faces (triangles): {len(faces)}")
                try:
                    coords, faces = mne.decimate_surface(coords, faces, n_triangles=len(faces) // 5)
                except Exception:
                    pass
                vertices_list.append(coords)
                faces_list.append(faces + vertex_offset)
                vertex_offset += len(coords)

        if vertices_list:
            all_vertices = np.vstack(vertices_list).flatten().tolist()
            all_faces = np.vstack(faces_list).flatten().tolist()

            # HTML template for interactive 3D figure visualization
            html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Interactive 3D Model - Sub-{subject}</title>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #111; font-family: sans-serif; }}
        #info {{
            position: absolute; top: 10px; left: 10px; color: white;
            background: rgba(0,0,0,0.7); padding: 10px 15px; border-radius: 8px; pointer-events: none;
        }}
    </style>
    <!-- Three.js y OrbitControls desde CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="info">
        <b>3D model ({fs_subject})</b><br>
        • Left-click: Rotate<br>
        • Mouse wheel: Zoom<br>
        • Right-click: Scroll
    </div>
    <script>
        const vertices = new Float32Array({json.dumps(all_vertices)});
        const indices = new Uint32Array({json.dumps(all_faces)});

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111116);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, -200, 50);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
        geometry.center();

        const material = new THREE.MeshPhongMaterial({{
            color: 0xd0d5dd,
            specular: 0x222222,
            shininess: 25,
            side: THREE.DoubleSide
        }});

        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);

        const ambientLight = new THREE.AmbientLight(0x666666);
        scene.add(ambientLight);

        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight1.position.set(1, 1, 1).normalize();
        scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0x555555, 0.5);
        dirLight2.position.set(-1, -1, -1).normalize();
        scene.add(dirLight2);

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>"""

            brain_html = interactive_dir / f'sub-{subject}_brain_3d.html'
            brain_html.write_text(html_content, encoding='utf-8')
            generated.append(('Interactive 3D Brain Surface', brain_html.name, 'iframe'))
            logger.info(f"3D model generated in {brain_html}")

    except Exception as e:
        logger.warning(f"The 3D surface could not be generated: {e}")

    # 3D INTERACTIVE MODEL (temporal animation)
    try:
        stc_candidates = sorted(deriv_root.rglob(f"sub-{subject}*+hemi.h5")) or \
                         sorted(deriv_root.rglob(f"sub-{subject}*-lh.stc"))
        if stc_candidates:
            stc_file = str(stc_candidates[0])
            if stc_file.endswith('-lh.stc'):
                stc_file = stc_file[:-len('-lh.stc')]
            stc = mne.read_source_estimate(stc_file)

            stc = stc.copy().decim(decim=8)


            subj_path = Path(subjects_dir) / fs_subject / 'surf'
            vertices_list, faces_list = [], []
            vertex_offset = 0
            for surf_name in ['lh.inflated', 'rh.inflated']:
                surf_path = subj_path / surf_name
                if not surf_path.exists():
                    surf_path = subj_path / surf_name.replace('inflated', 'pial') # fallback
                
                if surf_path.exists():
                    coords, faces = mne.read_surface(str(surf_path))
                    vertices_list.append(coords)
                    faces_list.append(faces + vertex_offset)
                    vertex_offset += len(coords)

            if vertices_list:
                all_vertices = np.vstack(vertices_list).flatten().tolist()
                all_faces = np.vstack(faces_list).flatten().tolist()
                n_lh_verts = len(vertices_list[0])

                # stc.data: [n_sources x n_times]; map sources -> surface via vertno
                time_data = stc.data.T.tolist()
                times = stc.times.tolist()
                lh_vertno = stc.lh_vertno.tolist()
                rh_vertno = stc.rh_vertno.tolist()
                n_lh_sources = len(lh_vertno)

                data_abs = np.abs(stc.data)
                vmin = float(np.min(data_abs[data_abs > 0])) if np.any(data_abs > 0) else 0.0
                vmax = float(np.percentile(data_abs, 95))
                if vmax <= vmin:
                    vmax = vmin + 1e-10

                html_time_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Actividad Temporal - Sub-{subject}</title>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #1a1a1a; font-family: sans-serif; }}
        #ui {{
            position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
            background: rgba(0,0,0,0.85); padding: 15px 25px; border-radius: 10px; color: white;
            text-align: center; width: 60%; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        }}
        #time-slider {{ width: 100%; margin-top: 10px; cursor: pointer; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 8px; pointer-events: none; }}
        #colorbar {{
            position: absolute; bottom: 90px; left: 20px; color: white;
            background: rgba(0,0,0,0.7); padding: 10px 12px; border-radius: 8px; pointer-events: none;
            font-size: 12px;
        }}
        #colorbar-gradient {{
            width: 180px; height: 14px; margin-bottom: 6px;
            background: linear-gradient(to right, #555, #ff0000, #ffff00, #ffffff);
            border-radius: 3px;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="info"><b>Estimación de Fuentes en el Tiempo</b><br>Visualizando activación dinámica</div>
    <div id="colorbar">
        <div id="colorbar-gradient"></div>
        <span>{vmin:.1f}</span> &nbsp;—&nbsp; <span>{vmax:.1f}</span>
    </div>
    <div id="ui">
        <div>Tiempo: <span id="time-display">{times[0]:.3f}</span> s</div>
        <input type="range" id="time-slider" min="0" max="{len(times)-1}" value="0" step="1">
    </div>
    <script>
        const vertices = new Float32Array({json.dumps(all_vertices)});
        const indices = new Uint32Array({json.dumps(all_faces)});
        const timeData = {json.dumps(time_data)};
        const times = {json.dumps(times)};
        const lhVertno = {json.dumps(lh_vertno)};
        const rhVertno = {json.dumps(rh_vertno)};
        const nLhVerts = {n_lh_verts};
        const nLhSources = {n_lh_sources};
        const vmin = {vmin};
        const vmax = {vmax};

        const BASE_R = 0.55, BASE_G = 0.55, BASE_B = 0.58;

        function hotColor(t) {{
            if (t <= 0) return [0.33, 0.33, 0.33];
            if (t < 0.375) return [t / 0.375, 0, 0];
            if (t < 0.75) return [1, (t - 0.375) / 0.375, 0];
            return [1, 1, (t - 0.75) / 0.25];
        }}

        function activationToColor(val) {{
            const intensity = Math.abs(val);
            if (intensity <= vmin) return [BASE_R, BASE_G, BASE_B];
            const t = Math.min(1.0, (intensity - vmin) / (vmax - vmin));
            const hot = hotColor(t);
            return [
                BASE_R + t * (hot[0] - BASE_R),
                BASE_G + t * (hot[1] - BASE_G),
                BASE_B + t * (hot[2] - BASE_B),
            ];
        }}

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111116);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, -200, 50);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
        geometry.center();

        const colors = new Float32Array(vertices.length);
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.MeshStandardMaterial({{
            vertexColors: true,
            side: THREE.DoubleSide,
            roughness: 0.5,
            metalness: 0.05
        }});

        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(1, 1, 1).normalize();
        scene.add(dirLight);

        function updateBrainColors(frameIndex) {{
            const activations = timeData[frameIndex];
            const colorAttr = geometry.getAttribute('color');
            const totalVertices = vertices.length / 3;

            for (let i = 0; i < totalVertices; i++) {{
                colorAttr.setXYZ(i, BASE_R, BASE_G, BASE_B);
            }}

            for (let s = 0; s < nLhSources; s++) {{
                const vertIdx = lhVertno[s];
                const [r, g, b] = activationToColor(activations[s]);
                colorAttr.setXYZ(vertIdx, r, g, b);
            }}

            for (let s = 0; s < rhVertno.length; s++) {{
                const vertIdx = nLhVerts + rhVertno[s];
                const [r, g, b] = activationToColor(activations[nLhSources + s]);
                colorAttr.setXYZ(vertIdx, r, g, b);
            }}

            colorAttr.needsUpdate = true;
            document.getElementById('time-display').innerText = times[frameIndex].toFixed(3);
        }}

        updateBrainColors(0);

        const slider = document.getElementById('time-slider');
        slider.addEventListener('input', (e) => {{
            updateBrainColors(parseInt(e.target.value));
        }});

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>"""

                time_html_path = interactive_dir / f'sub-{subject}_source_animation.html'
                time_html_path.write_text(html_time_content, encoding='utf-8')
                generated.append(('Interactive Neural Activity Over Time', time_html_path.name, 'iframe'))
                logger.info(f"Temporal source animation generated in {time_html_path}")
    except Exception as e:
        logger.warning(f"The temporal interactive source animation could not be generated: {e}")

    # HTML index
    if generated:
        index_path = interactive_dir / 'index.html'
        with open(index_path, 'w', encoding='utf-8') as idx:
            idx.write("<html><head><meta charset='utf-8'>"
                       "<title></title>Interactive 3D Visualizations</head><body>")
            idx.write(f"<h1>Sub-{subject}: Interactive 3D Visualizations</h1>")
            idx.write("<p>Drag with the left mouse button to rotate and scroll the mouse wheel to zoom</p>")
            
            for label, filename, kind in generated:
                idx.write(f"<div style='background:white; padding:15px; margin-bottom:25px; border-radius:8px; box-shadow:0 2px 5px rgba(0,0,0,0.1);'>")
                idx.write(f"<h2 style='color:#0277bd; margin-top:0;'>{label}</h2>")
                if kind == 'iframe':
                    idx.write(f"<iframe src='{filename}' width='100%' height='600' style='border:none; border-radius:4px;'></iframe>")
                else:
                    idx.write(f"<div style='text-align:center;'><img src='{filename}' style='max-width:100%; height:auto; border-radius:4px;'/></div>")
                idx.write(f"</div>")
                
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

    fs_path = str(original_fs_home.resolve())
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

if steps is not None:
    command = ["mne_bids_pipeline", f"--config={file_name}", f"--steps={steps}"]
    try:
        subprocess.run(command, check=True, env=os.environ.copy())
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

for path in real_deriv_root.rglob("*.html"):
    if "sub-average" not in path.name:
        logger.info(f"{path.name} copied to the output")
        dest = html_report_dir/path.name
        copyfile(path, dest)

