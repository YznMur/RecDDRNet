#!/bin/bash
set -euo pipefail

docker exec --user trainer -it recddrnet \
    /bin/bash -c "
    cd /home/trainer/RecDDRNet;
    nvidia-smi;
    /bin/bash"
