# Toolchain for building Cinnamon's PuD path: LLVM/MLIR 19.1.3, cinm-opt, and
# the kernels' LLVM stages. The image holds only the tools; LLVM and Cinnamon
# are built into the bind-mounted source tree, so the build products live on
# the host rather than in Docker's storage.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang lld cmake ninja-build git curl ca-certificates \
      python3 zlib1g-dev libzstd-dev pkg-config \
      coinor-libcbc-dev \
 && rm -rf /var/lib/apt/lists/*

# lime and eggmock are Rust crates, pulled in through Corrosion. The toolchain
# is left world-writable so the container can run as the host user.
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path \
 && chmod -R a+rwX /opt/rustup /opt/cargo

# The mapping pass links Gurobi from this fixed path
# (cinnamon/lib/Dialect/Bits/Transforms/CMakeLists.txt). The licence is a
# run-time matter: mount it and point GRB_LICENSE_FILE at it.
RUN curl -sSfL https://packages.gurobi.com/13.0/gurobi13.0.0_linux64.tar.gz \
    | tar -xz -C /opt \
 && rm -rf /opt/gurobi1300/linux64/docs /opt/gurobi1300/linux64/examples \
           /opt/gurobi1300/linux64/matlab /opt/gurobi1300/linux64/R
ENV GUROBI_HOME=/opt/gurobi1300/linux64 \
    LD_LIBRARY_PATH=/opt/gurobi1300/linux64/lib \
    CC=clang CXX=clang++ CMAKE_GENERATOR=Ninja HOME=/tmp

