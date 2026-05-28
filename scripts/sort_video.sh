#!/bin/bash
set -euo pipefail

SRC_DIR="/home/trainer/data/video"

# Ensure script runs as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run with sudo"
  exit 1
fi

cd "$SRC_DIR"

for file in camera_*.mp4; do
    [ -e "$file" ] || continue

    cam_id=$(echo "$file" | cut -d'_' -f2)
    dir="${SRC_DIR}/camera_${cam_id}"

    mkdir -p "$dir"
    mv "$file" "$dir/"
done

echo "Sorting complete in $SRC_DIR"