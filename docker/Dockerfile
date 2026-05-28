# --------------------------
# Base image (PyTorch + CUDA + cuDNN)
# --------------------------
ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# --------------------------
# Fix networking issues (IPv4 + faster mirror)
# --------------------------
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirror.nl.leaseweb.net/ubuntu|g' /etc/apt/sources.list

# --------------------------
# System dependencies with fixed NVIDIA GPG key
# --------------------------
RUN apt-get update || true

# Remove any existing NVIDIA repositories to avoid conflicts
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# Install required tools and add NVIDIA GPG key
RUN apt-get install -y --no-install-recommends wget gnupg2 ca-certificates && \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# --------------------------
# Install system dependencies
# --------------------------
RUN apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --no-install-recommends --fix-missing \
        git \
        ninja-build \
        libglib2.0-0 \
        libsm6 \
        libxrender-dev \
        libxext6 \
        libgl1-mesa-glx \
        sudo \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# --------------------------
# Clone DDRNet repository
# --------------------------
RUN git clone https://github.com/chenjun2hao/DDRNet.pytorch.git /DDRNet.pytorch

# --------------------------
# Install Python dependencies
# --------------------------
RUN pip install --no-cache-dir \
    opencv-python==4.5.4.58 \
    EasyDict \
    shapely \
    Cython \
    scipy \
    pandas \
    pyyaml \
    json_tricks \
    scikit-image \
    yacs \
    tensorboardX \
    tqdm \
    ninja \
    colour-demosaicing \
    colour-science \
    clearml \
    numpy==1.19.5
RUN apt-get update && apt-get install -y openssh-client && rm -rf /var/lib/apt/lists/*
# --------------------------
# Add non-root user with host UID/GID support
# --------------------------
ARG USERNAME=trainer
ARG UID=1000
ARG GID=1000

RUN if getent group ${GID} >/dev/null; then \
        groupmod -n ${USERNAME} "$(getent group ${GID} | cut -d: -f1)"; \
    else \
        groupadd -g ${GID} ${USERNAME}; \
    fi && \
    if id -u ${UID} >/dev/null 2>&1; then \
        usermod -l ${USERNAME} -d /home/${USERNAME} -m "$(getent passwd ${UID} | cut -d: -f1)"; \
        usermod -g ${GID} ${USERNAME}; \
    else \
        useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}; \
    fi && \
    usermod -aG sudo ${USERNAME} && \
    echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# --------------------------
# Fix permissions (CRITICAL)
# --------------------------
RUN chown -R ${UID}:${GID} /DDRNet.pytorch && \
    mkdir -p /home/${USERNAME}/DDRNet.pytorch && \
    chown -R ${UID}:${GID} /home/${USERNAME}

# Optional: better default permissions
RUN echo "umask 002" >> /home/${USERNAME}/.bashrc

# --------------------------
# Switch to non-root user
# --------------------------
USER ${USERNAME}
WORKDIR /home/${USERNAME}
