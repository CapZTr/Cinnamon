# syntax=docker/dockerfile:1.7

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Required OS packages.
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    clang \
    coinor-cbc \
    coinor-libcbc-dev \
    lld \
    cmake \
    ninja-build \
    git \
    gdb \
    curl \
    wget \
    pkg-config \
    bash-completion \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Setup the virtual environment.
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install "pybind11<3" numpy pytest build

# Install Rust.
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

# Install the `just` tool.
RUN arch="$(dpkg --print-architecture)" && \
    case "$arch" in \
      amd64) JUST_ARCH=x86_64-unknown-linux-musl ;; \
      arm64) JUST_ARCH=aarch64-unknown-linux-musl ;; \
      *) echo "Unsupported arch: $arch" && exit 1 ;; \
    esac && \
    wget -qO- "https://github.com/casey/just/releases/download/1.51.0/just-1.51.0-${JUST_ARCH}.tar.gz" \
      | tar xz -C /usr/local/bin just

# Install UPMEM.
RUN wget \
    https://github.com/kagandikmen/upmem-sdk/raw/refs/heads/master/2024.2.0/upmem-2024.2.0-Linux-x86_64.tar.gz \
    -O upmem.tar.gz && \
    mkdir -p /opt/upmem && \
    tar xf upmem.tar.gz -C /opt/upmem --strip-components=1 && \
    rm upmem.tar.gz

# Setup .bashrc
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
RUN echo 'source /opt/venv/bin/activate' >> $HOME/.bashrc

# Setup environment variables.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/usr/local/bin:${PATH}
ENV GUROBI_ROOT=/opt/gurobi1302/linux64
ENV UPMEM_HOME=/opt/upmem

WORKDIR /workspace/cinnamon
CMD ["/bin/bash"]
