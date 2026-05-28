
## Main commands:

```bash
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/cityscapes/ddrnet23.yaml
python ./tools/train.py --cfg /home/trainer/DDRNet.pytorch/experiments/cityscapes/ddrnet23.yaml 
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg /home/trainer/DDRNet.pytorch/experiments/cityscapes/ddrnet23.yaml

```

