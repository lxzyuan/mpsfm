ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.8.0

# Builder stage for installing & compiling dependencies
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

ARG CUDA_ARCHITECTURES=native
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages & dependencies
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.init7.net/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirror.init7.net/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        cmake \
        ninja-build build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libgflags-dev \
        libatlas-base-dev \
        libsuitesparse-dev

# Build Ceres & pyceres
RUN git clone --branch 2.2.0 --depth 1 https://ceres-solver.googlesource.com/ceres-solver /ceres-solver && \
    mkdir /ceres-solver/build && cd /ceres-solver/build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja install
RUN git clone --depth 1 https://github.com/cvg/pyceres.git /pyceres && \
    python3 -m pip install /pyceres

# Install ruff & COLMAP
RUN python3 -m pip install ruff
ARG COLMAP_GIT_COMMIT=main
RUN git clone --depth 1 https://github.com/Zador-Pataki/colmap.git /colmap && \
    cd /colmap && git checkout ${COLMAP_GIT_COMMIT} && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja install && \
    python3 -m pip install /colmap

# Clean up to reduce image size
RUN rm -rf \
    /colmap \
    /ceres-solver \
    /pyceres \
    /root/.cache \
    /usr/local/include \
    /usr/local/lib/cmake \
    /usr/local/share \
    /usr/local/lib/*.a \
    /usr/local/lib/*.la

# Runtime stage for minimal runtime dependencies
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime
ENV DEBIAN_FRONTEND=noninteractive

# Install only what's needed at runtime
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.init7.net/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirror.init7.net/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        git \ 
        wget \
        libboost-program-options-dev \
        libatlas-base-dev \
        libceres-dev libfreeimage-dev libglew-dev libgoogle-glog-dev \
        libqt5core5a libqt5gui5 libqt5widgets5 libcurl4 \
        libopenblas0-pthread

# Copy compiled artifacts from builder
COPY --from=builder /usr/local/ /usr/local/
ENV PATH=/usr/local/bin:$PATH

# Install Python requirements
WORKDIR /mpsfm
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    rm -rf /root/.cache

# Final entrypoint
ENTRYPOINT ["bash"]
