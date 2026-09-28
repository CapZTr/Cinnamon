# syntax=docker/dockerfile:1.4
#
# PuD tutorial image: Cinnamon's compiled processing-using-DRAM path (LLVM/MLIR
# 19.1.3, cinm-opt) + MIMDRAM's gem5 + Jupyter. Binder-compatible: user jovyan,
# UID 1000, the repository in $HOME, nothing needs Docker at run time.
#
# Build with tutorial/build-image.sh, which passes MIMDRAM's committed tree as
# the `mimdram` build context. The mapping pass needs a Gurobi licence, which
# is not in the image: readers put their own gurobi.lic in $HOME.

############################################################################
# Stage 1: gem5 on the userland it was written for
############################################################################
# MIMDRAM's gem5 is tested on Ubuntu 14.04 (Python 2.7, scons 2, gcc 4.8), and
# the programs it runs have to be linked there too: a static binary from a
# current toolchain does not get past its first instructions under this gem5's
# syscall emulation. Both the simulator and that toolchain are carried into the
# final image from here, in /usr, which runs unchanged on the newer glibc.
FROM ubuntu:14.04 AS gem5
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git m4 scons zlib1g zlib1g-dev \
      libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
      python-dev python swig \
 && rm -rf /var/lib/apt/lists/*

COPY --from=mimdram . /src/MIMDRAM
ARG JOBS=8
RUN cd /src/MIMDRAM/gem5 && scons build/X86_PUD/gem5.opt -j "${JOBS}"

# Export: the tree without build intermediates, plus the one binary.
RUN mkdir -p /export \
 && tar -C /src --exclude='MIMDRAM/gem5/build' -cf - MIMDRAM | tar -C /export -xf - \
 && mkdir -p /export/MIMDRAM/gem5/build/X86_PUD \
 && cp /src/MIMDRAM/gem5/build/X86_PUD/gem5.opt /export/MIMDRAM/gem5/build/X86_PUD/

############################################################################
# Stage 2: the compiler, the notebooks, Jupyter
############################################################################
FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang lld cmake ninja-build git curl ca-certificates \
      python3 python3-venv python3-dev zlib1g-dev libzstd-dev pkg-config \
      coinor-libcbc-dev \
 && rm -rf /var/lib/apt/lists/*

# Binder: UID 1000 (ubuntu:24.04 already has user 'ubuntu' with that UID).
ARG NB_USER=jovyan
ARG NB_UID=1000
RUN userdel -r ubuntu && useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER}
ENV USER=${NB_USER} HOME=/home/${NB_USER}

# Gurobi: the library the mapping pass links, at the path its CMake names. No
# licence: that is the reader's.
RUN curl -sSfL https://packages.gurobi.com/13.0/gurobi13.0.0_linux64.tar.gz \
    | tar -xz -C /opt \
 && rm -rf /opt/gurobi1300/linux64/{docs,examples,matlab,R,src}

# LLVM/MLIR 19.1.3 with the repository's patch, installed and the build tree
# dropped in the same step, so the layer holds only what is used.
COPY patches/llvm /tmp/llvm-patches
ARG JOBS=8
RUN git clone -q --depth 1 --branch llvmorg-19.1.3 \
      https://github.com/llvm/llvm-project /tmp/llvm-project \
 && cd /tmp/llvm-project \
 && for p in /tmp/llvm-patches/*.patch; do git apply "$p"; done \
 && cmake -S llvm -B build -G Ninja -Wno-dev \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/llvm \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld \
      -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=X86 \
      -DLLVM_ENABLE_ASSERTIONS=ON -DBUILD_SHARED_LIBS=ON \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_OPTIMIZED_TABLEGEN=ON \
      -DLLVM_PARALLEL_LINK_JOBS=4 \
 && ninja -C build -j "${JOBS}" install \
 && rm -rf /tmp/llvm-project /tmp/llvm-patches

# The 14.04 userland and gem5 from stage 1, and the two commands the notebooks
# use them through (docker/tutorial/): each sets the library path for itself
# only, since exported it would put 14.04's libstdc++ under everything else.
COPY --from=gem5 /usr /opt/trusty/usr
COPY --from=gem5 --chown=${NB_UID} /export/MIMDRAM ${HOME}/third-party/MIMDRAM
COPY docker/tutorial/pud-gcc docker/tutorial/gem5-pud /usr/local/bin/

# The compiler's sources first and the rest of the repository after it, so that
# editing a notebook does not rebuild cinm-opt.
COPY --chown=${NB_UID} cinnamon ${HOME}/cinnamon
USER ${NB_USER}
WORKDIR ${HOME}

# cinm-opt. Rust (for the lime and eggmock crates) is installed, used and
# removed in this one step, and only the binaries are kept of the build tree.
RUN export RUSTUP_HOME=/tmp/rustup CARGO_HOME=/tmp/cargo \
 && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path \
 && export PATH=/tmp/cargo/bin:$PATH \
 && cmake -S cinnamon -B cinnamon/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld \
      -DLLVM_DIR=/opt/llvm/lib/cmake/llvm -DMLIR_DIR=/opt/llvm/lib/cmake/mlir \
      -DCINM_BUILD_GPU_SUPPORT=OFF -DUPMEM_DIR=OFF -DTORCH_MLIR_DIR=OFF \
 && ninja -C cinnamon/build -j "${JOBS}" cinm-opt rowop-sanity \
 && find cinnamon/build -mindepth 1 -maxdepth 1 ! -name bin -exec rm -rf {} + \
 && rm -rf /tmp/rustup /tmp/cargo

# Jupyter (required by Binder) and what the notebooks plot with.
RUN python3 -m venv ${HOME}/.venv \
 && ${HOME}/.venv/bin/pip install --no-cache-dir notebook jupyterlab matplotlib

# The rest of the repository: the notebooks, the kernels, this file.
COPY --chown=${NB_UID} . ${HOME}

# What a shell would otherwise have to set up: `jupyter` is started directly,
# here and on Binder. Last, so the builds above use the system toolchain.
ENV CINM_ROOT=${HOME} \
    MIMDRAM_ROOT=${HOME}/third-party/MIMDRAM \
    VIRTUAL_ENV=${HOME}/.venv \
    PATH=${HOME}/.venv/bin:${HOME}/cinnamon/build/bin:/opt/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/llvm/lib:/opt/gurobi1300/linux64/lib
