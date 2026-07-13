FROM python:3.11-slim
 
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MNE_BROWSER_BACKEND=matplotlib \
    MPLBACKEND=Agg \
    PYVISTA_OFF_SCREEN=true \
    PYVISTA_USE_PANEL=false \
    MNE_3D_OPTION_ANTIALIAS=false \
    LIBGL_ALWAYS_SOFTWARE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1
 
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libgl1-mesa-dri \
        libosmesa6 \
        libegl1 \
        xvfb \
        xauth \
        libglib2.0-0 \
        curl \
        tcsh \
        bc \
        tar \
        gzip \
        unzip \
        libgomp1 \
        libgsl-dev \
        libx11-6 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libdbus-1-3 \
        libxcb1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libice6 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libfontconfig1 \
        libfreetype6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
 
RUN pip install --no-cache-dir \
        "numpy<2" \
        "scipy" \
        "matplotlib" \
        "scikit-learn" \
        mne \
        mne-bids \
        mne-bids-pipeline==1.10.1 \
        pyvista \
        pyvistaqt \
        PyQt5 \
    && find /usr/local/lib/python3.11 -type d -name "__pycache__" -exec rm -rf {} + \
    && find /usr/local/lib/python3.11 -type d \( -name "tests" -o -name "test" \) -exec rm -rf {} + \
    && rm -rf /root/.cache /tmp/*
 
RUN curl -fsSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz \
    | tar -xz -C /opt && \
    rm -rf \
        /opt/freesurfer/subjects/fsaverage3 \
        /opt/freesurfer/subjects/fsaverage4 \
        /opt/freesurfer/subjects/fsaverage5 \
        /opt/freesurfer/subjects/fsaverage6 \
        /opt/freesurfer/subjects/cv90 \
        /opt/freesurfer/subjects/bert \
        /opt/freesurfer/subjects/sample \
        /opt/freesurfer/subjects/V1_average \
        /opt/freesurfer/subjects/fsaverage_sym \
        /opt/freesurfer/subjects/cvs_avg35 \
        /opt/freesurfer/subjects/cvs_avg35_inMNI152 \
        /opt/freesurfer/trctrain \
        /opt/freesurfer/fsfast \
        /opt/freesurfer/matlab \
        /opt/freesurfer/docs \
        /opt/freesurfer/average
 
WORKDIR /work
 
RUN rm -f /bin/sh && ln -s /bin/bash /bin/sh
 
RUN ldconfig
 
ENV FREESURFER_HOME=/opt/freesurfer \
    SUBJECTS_DIR=/opt/freesurfer/subjects \
    PATH=/opt/freesurfer/bin:/opt/freesurfer/fsfast/bin:/opt/freesurfer/tktools:/opt/freesurfer/mni/bin:$PATH
