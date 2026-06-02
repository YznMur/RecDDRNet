#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

docker build \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  -t recddrnet:latest \
  -f Dockerfile \
  ..
