#!/bin/bash
python tools/eval.py --cfg experiments/rsm/ddrnet23_slim_640x480_5classes_temporal_attention.yaml \
    --splits train,val \
    DATASET.TRAIN_SET list/rsm/train_RSMdataset_v2_seq30.lst \
    DATASET.TEST_SET list/rsm/val_RSMdataset_v2_seq30.lst