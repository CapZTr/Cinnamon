#!/usr/bin/env bash
# Run a command in the cinnamon-pud toolchain container, as you, with the
# workspace (Cinnamon and MIMDRAM side by side) mounted at the same path it has
# on the host. The Docker socket is passed through so that
# compile_pud_binary.sh can link and simulate in the `mimdram` container.
#
#   docker/run.sh ninja -C cinnamon/build-pud cinm-opt
#   docker/run.sh                                  # an interactive shell
set -euo pipefail

WS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# The mapping pass needs a Gurobi licence usable from the C++ API (not the
# pip one). Point GRB_LICENSE_FILE at it.
LIC=${GRB_LICENSE_FILE:-}
[[ -n $LIC ]] || echo "run.sh: GRB_LICENSE_FILE not set; --bits-opt-mapping will fail" >&2

args=(--rm -i --user "$(id -u):$(id -g)" --group-add "$(getent group docker | cut -d: -f3)"
      -v "$WS:$WS" -w "$PWD"
      -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker:ro
      -e CIMDRAM_ROOT="$WS")
[[ -n $LIC && -f $LIC ]] && args+=(-v "$LIC:/opt/gurobi.lic:ro" -e GRB_LICENSE_FILE=/opt/gurobi.lic)
[[ -t 0 ]] && args+=(-t)

exec docker run "${args[@]}" cinnamon-pud "${@:-bash}"
