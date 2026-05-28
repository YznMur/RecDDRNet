#!/bin/bash
set -e

echo "====================================="
echo " DDRNet ROOT ONLY CLEANUP"
echo "====================================="

mkdir -p scripts
mkdir -p docs
mkdir -p reports
mkdir -p lists
mkdir -p runs
mkdir -p deploy
mkdir -p media_tools
mkdir -p models

echo "[1] Moving shell scripts..."
for f in train.sh onnx.sh onnx_single_image.sh sort_video.sh vcutter_preview.sh video_inf.sh; do
    [ -f "$f" ] && mv "$f" scripts/
done

echo "[2] Moving python media/helper scripts..."
for f in create_preview.py preview.py make_list.py report.py videos_to_frames.py two_videos.py two_videos_onnx_1.py two_videos_onnx_2.py two_videos_onnx_3.py; do
    [ -f "$f" ] && mv "$f" media_tools/
done

echo "[3] Moving deployment/inference scripts..."
for f in inference_trt_ddrnet.py; do
    [ -f "$f" ] && mv "$f" deploy/
done

echo "[4] Moving docs and references..."
mv commands.md docs/ 2>/dev/null || true
mv dependencies.txt docs/ 2>/dev/null || true
mv key.txt docs/ 2>/dev/null || true
mv *.xlsx docs/ 2>/dev/null || true

echo "[5] Moving pdf reports..."
mv *.pdf reports/ 2>/dev/null || true

echo "[6] Moving dataset list files..."
mv *.lst lists/ 2>/dev/null || true

echo "[7] Moving model exports..."
mv *.engine models/ 2>/dev/null || true

echo "[8] Moving loose experiment/result folders..."
for d in \
cityscapes_ddrnet23_slim_640x480 \
DDRNet23_RSM_v2_imagenet_2048_super_classes \
RSM_v2_imagenet_2048_super_classes \
RSM_v2_imagenet_2048_super_classes_aug_lr_005 \
RSM_v2_imagenet_2048_super_classes_aug_lr_v2_512 \
RSM_v2_imagenet_2048_super_classes_aug_lr_v3_1024 \
RSM_v2_imagenet_2048_super_classes_aug_lr_v3_512 \
RSM_v2_rsm_640x480_5_super_classes \
RSM_v2_rsm_640x480_5_super_classes_weighted \
RSM_v2_rsm_640x480_5_super_classes_weighted_CombinedLoss \
RSM_v2_rsm_640x480_super_classes \
RSM_v3_aug_stronger_photometric_poly \
rsm_ddrnet23_output \
output_rsm_ddrnet39
do
    [ -d "$d" ] && mv "$d" runs/
done

echo "[9] Moving pretrained models..."
[ -d pretrained_models ] && mv pretrained_models models/

echo "====================================="
echo " ROOT CLEANUP COMPLETE"
echo "====================================="