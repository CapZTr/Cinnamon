# Tutorial: compiling for processing-using-DRAM

Hands-on notebooks for Cinnamon's processing-using-DRAM (PuD) path: a `linalg`
kernel is compiled into an x86 program in which every DRAM command is one
instruction, and MIMDRAM's gem5 runs it on a processor and a DRAM model together.

| notebook | |
|---|---|
| `00_intro` | what PuD is, what is in the image, the Gurobi licence |
| `01_vadd_pipeline` | `vadd` through every compiler stage, reading each intermediate |
| `02_vadd_run` | link and run it on gem5; check the command count by hand |
| `03_bank_sweep` | 8 → 64 banks: the mapping, the speed-up, and the second channel |
| `04_compiled_vs_trace` | the compiled program against the paper's trace-driven simulation |
| `05_reduction` | `gemv`: how partial results from 64 banks are combined |
| `06_lime` | lime on its own: a circuit written with mockturtle, compiled with `lime_program_export` and its arguments |
| `07_primitives_and_approximate_add` | redefine a primitive in MLIR (`--pud-primitives`); an approximate adder (`--pud-approx-add-low-bits`) and its cost against its error |

Each gem5 run takes seconds to a few minutes and under 300 MB of memory.

## You need a Gurobi licence

The mapping step solves an ILP with Gurobi, which is in the image; a licence is
not. Use a free **academic WLS licence** (named-user licences are locked to one
machine, and the one `pip install gurobipy` brings only works from Python).
Notebook 00 explains how to get it and checks that it works: put `gurobi.lic` in
your home directory.

## Running it

**On Binder:** open the repository on [mybinder.org](https://mybinder.org) with
*File* `tutorial/notebooks/00_intro.ipynb`. `binder/Dockerfile` points Binder at
the prebuilt image, so nothing is compiled there.

**Locally:**

```sh
docker run -it --rm -p 8888:8888 \
  -v /path/to/gurobi.lic:/home/jovyan/gurobi.lic:ro \
  ghcr.io/joaopaulocl/cinnamon-pud-tutorial:esweek \
  jupyter lab --ip=0.0.0.0 --port=8888
```

## Building and publishing the image

```sh
tutorial/build-image.sh cinnamon-pud-tutorial:esweek
```

The script expects MIMDRAM (branch `pud-codegen`) next to this repository, or at
`$MIMDRAM`, and builds from its **committed** HEAD. Stage 1 builds gem5 on Ubuntu
14.04; stage 2 builds LLVM 19.1.3 and `cinm-opt` on 24.04 and installs Jupyter.
A first build takes about an hour, mostly LLVM; changing only notebooks rebuilds
the last layers in seconds.

The image carries Ubuntu 14.04's `/usr` in `/opt/trusty`, used through two
commands (`docker/tutorial/`):

- `pud-gcc` links kernels with gcc 4.8 and 14.04's C library. This gem5's syscall
  emulation cannot run the startup code of a current C library: a gcc 13 static
  binary fails before its first system call.
- `gem5-pud` runs gem5 with the libraries and Python 2.7 it was built against.

Test inside the image before pushing, in notebook order:

```sh
docker run --rm -v /path/to/gurobi.lic:/home/jovyan/gurobi.lic:ro \
  cinnamon-pud-tutorial:esweek bash -c 'cd ~/tutorial/notebooks && \
  for nb in 00_intro 01_vadd_pipeline 02_vadd_run 03_bank_sweep 04_compiled_vs_trace 05_reduction 06_lime 07_primitives_and_approximate_add; do
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 \
      --output /tmp/$nb.out.ipynb $nb.ipynb >/tmp/$nb.log 2>&1 \
      && echo "OK   $nb" || { echo "FAIL $nb"; tail -15 /tmp/$nb.log; }
  done'
```

Publishing:

```sh
docker tag cinnamon-pud-tutorial:esweek ghcr.io/joaopaulocl/cinnamon-pud-tutorial:esweek
docker push ghcr.io/joaopaulocl/cinnamon-pud-tutorial:esweek
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/joaopaulocl/cinnamon-pud-tutorial:esweek
```

Then put that digest in `binder/Dockerfile`: every rebuild changes it.

## Layout

- `notebooks/` — the notebooks.
- `_infra/pud.py` — one function per compiler stage, plus running gem5 and the
  trace player. Tools are found through environment variables first
  (`CINM_OPT`, `LLVM_BIN`, `PUD_CC`, `GEM5_PUD`, `MIMDRAM_ROOT`), so the same
  notebooks run outside the image too.
- `artifacts/` — everything the notebooks write, generated when they run.
