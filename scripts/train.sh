# python tools/train.py --cfg experiments/rsm/ddrnet23_slim_640x480_5classes.yaml
# python tools/train.py --cfg experiments/rsm/ddrnet23_slim_updated.yaml
# python tools/train.py --cfg experiments/rsm/ddrnet23_updated.yaml
# python tools/train.py --cfg experiments/cityscapes/ddrnet23.yaml
# python tools/train.py --cfg experiments/rsm/ddrnet39.yaml
# python tools/train.py --cfg configs/rsm_ddrnet23slim_convlstm.yaml
# python tools/train.py --cfg experiments/rsm/ddrnet23_slim_640x480_5classes.yaml MODEL.NAME ddrnet23_slim_convlstm
# python tools/eval.py --cfg experiments/rsm/ddrnet23_slim_640x480_5classes.yaml
# python tools/demo.py --cfg configs/rsm_ddrnet23slim_convlstm.yaml
python tools/train.py --cfg configs/rsm_ddrnet23slim_convlstm.yaml
