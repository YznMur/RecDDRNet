#!/bin/bash
set -euo pipefail

# Move to project root
cd "$(dirname "$0")"
cd ..

workspace_dir=$PWD

# Get host user UID/GID
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Remove existing container if it exists (running or stopped)
if [ "$(docker ps -aq -f name=recddrnet)" ]; then
    docker rm -f recddrnet
fi

docker_args=(
    -it -d --rm
    --gpus all
    --net host
    --user ${USER_ID}:${GROUP_ID}
    -e HOME=/home/trainer
    -e HOST_UID=${USER_ID}
    -e HOST_GID=${GROUP_ID}
    -e NVIDIA_DRIVER_CAPABILITIES=all
    --shm-size=45g
    --name recddrnet
    -v "$workspace_dir/":/home/trainer/RecDDRNet/:rw
    -v /home/ymurhij/datasets/RSMcityscapes_5_superclasses_v3:/home/trainer/RecDDRNet/data/rsm:rw
    -v /home/ymurhij/datasets/Cityscapes:/home/trainer/RecDDRNet/data/cityscapes:rw
    -v /home/ymurhij/.clearml/clearml.conf:/home/trainer/.clearml/clearml.conf:rw
    -v /home/ymurhij/.clearml:/home/trainer/.clearml
    -w /home/trainer/RecDDRNet
    # -v /home/ymurhij/preprocess/data/output:/home/trainer/data/video:rw
)

# Allow X11 access only when a display is available
if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null 2>&1; then
    xhost +local:docker
    docker_args+=(
        -e DISPLAY=${DISPLAY}
        -e QT_X11_NO_MITSHM=1
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    )
else
    echo "DISPLAY is not set; starting container without X11 forwarding."
fi

docker run "${docker_args[@]}" recddrnet:latest
