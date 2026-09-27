# MIMDRAM's gem5 and the final link of a compiled kernel. Ubuntu 14.04 is what
# MIMDRAM was tested on, and its gcc 4.8 / static glibc stays inside what gem5's
# syscall emulation supports. Run it with MIMDRAM mounted at /workspace and
# named `mimdram`, which is what compile_pud_binary.sh expects.
FROM ubuntu:14.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git m4 scons zlib1g zlib1g-dev \
      libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
      python-dev python swig \
 && rm -rf /var/lib/apt/lists/*

ENV HOME=/tmp
WORKDIR /workspace
