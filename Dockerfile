FROM python:3.11-slim 
 
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MNE_BROWSER_BACKEND=matplotlib \
    MPLBACKEND=Agg \
    PYVISTA_OFF_SCREEN=true \
    MNE_3D_OPTION_ANTIALIAS=false \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MNE_3D_BACKEND=pyvistaqt \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1
 
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        git \
        libgl1 \
        libgl1-mesa-dri \
        libosmesa6 \
        libegl1 \
        libglib2.0-0 \
        curl \
        tcsh \
        bc \
        tar \
        gzip \
        unzip \
        libgomp1 \
        libgsl-dev \
        libquadmath0 \
        libgfortran5 \
        libxmu6 \
        libxt6 \
        libxext6 \
        perl \
        libjpeg62-turbo \
        binutils \
        psmisc \
        libfontconfig1 \
        libfreetype6 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libdbus-1-3 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-sync1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libx11-xcb1 \
        libsm6 \
        libice6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
 
RUN pip install --no-cache-dir \
        "numpy<2" \
        "scipy" \
        "matplotlib" \
        "scikit-learn" \
        "PyQt5" \
        pyvista==0.46.3 \
        pyvistaqt==0.11.3 \
        nest_asyncio \
        ipyevents \
        ipywidgets \
        trame \
        trame-vtk \
        trame-vuetify \
        mne \
        mne-bids \
        mne-bids-pipeline==1.10.1 \
    && pip uninstall -y vtk \
    && pip install --no-cache-dir --extra-index-url https://wheels.vtk.org vtk-osmesa \
    && find /usr/local/lib/python3.11 -type d -name "__pycache__" -exec rm -rf {} + \
    && find /usr/local/lib/python3.11 -type d \( -name "tests" -o -name "test" \) -exec rm -rf {} + \
    && rm -rf /root/.cache /tmp/*

RUN curl -fsSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz \
    | tar -xz -C /opt

WORKDIR /work
 
RUN rm -f /bin/sh && ln -s /bin/bash /bin/sh
 
RUN ldconfig
 
ENV FREESURFER_HOME=/opt/freesurfer \
    SUBJECTS_DIR=/opt/freesurfer/subjects \
    PATH=/opt/freesurfer/bin:/opt/freesurfer/fsfast/bin:/opt/freesurfer/tktools:/opt/freesurfer/mni/bin:$PATH