python ./tools/onnx_inf_single_image.py \
  --model output/ddrnet_23_slim_fp32_bs1.onnx \
  --image cv_cap.png \
  --output out_2 \
  --num_classes 4  \
  --input_size 480,640

    # --input_size 512,1024