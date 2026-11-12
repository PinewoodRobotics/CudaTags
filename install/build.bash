#!/usr/bin/env bash
# Fresh Jetson setup: dependencies, then build, test and install CudaTags.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$HERE/deps.bash"
bash "$HERE/self.bash"
