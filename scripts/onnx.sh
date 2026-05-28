# python ./tools/onnx_inf.py \
#   --model rsm_ddrnet23_output/ddrnet_23_fp32_bs1.onnx \
#   --image /home/trainer/DDRNet.pytorch/data/rsm/leftImg8bit/test/ \
#   --gt_root /home/trainer/DDRNet.pytorch/data/rsm/gtFine \
#   --split test \
#   --output output_rsm_v2/onnx_output_results_test_rsm_v2 \
#   --num_classes 4 \
#   --input_size 480,640

# python ./tools/onnx_inf.py \
#   --model rsm_ddrnet23_output/ddrnet_23_fp32_bs1.onnx \
#   --image /home/trainer/DDRNet.pytorch/data/rsm/leftImg8bit/val/ \
#   --gt_root /home/trainer/DDRNet.pytorch/data/rsm/gtFine \
#   --split val \
#   --output output_rsm_v2/onnx_output_results_val_rsm_v2 \
#   --num_classes 4 \
#   --input_size 480,640

python ./tools/onnx_inf.py \
  --model rsm_ddrnet23_output/ddrnet_23_fp32_bs1.onnx \
  --image /home/trainer/DDRNet.pytorch/data/rsm/leftImg8bit/train/ \
  --gt_root /home/trainer/DDRNet.pytorch/data/rsm/gtFine \
  --split train \
  --output output_rsm_v2/onnx_output_results_train_rsm_v2 \
  --num_classes 4 \
  --input_size 480,640
