# mybinder.org: start from the prebuilt tutorial image instead of rebuilding
# LLVM (see ../Dockerfile and tutorial/build-image.sh). The image already has
# the repository in $HOME, Jupyter, and runs as UID 1000.
#
# Update the digest after every push:
#   docker inspect --format='{{index .RepoDigests 0}}' <image>
FROM ghcr.io/joaopaulocl/cinnamon-pud-tutorial@sha256:REPLACE_AFTER_PUSH
