#!/bin/bash
set -euo pipefail

docker exec --user trainer -it ddrnetpy \
    /bin/bash -c "
    cd /home/trainer/DDRNet.pytorch;
    nvidia-smi;
    /bin/bash"
