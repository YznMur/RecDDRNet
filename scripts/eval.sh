#!/bin/bash
python tools/eval.py --cfg configs/rsm_ddrnet23slim_temporal_attn.yaml \
    --splits train,val \
    DATASET.TRAIN_SET list/rsm/train_RSMdataset_v2_seq30.lst \
    DATASET.TEST_SET list/rsm/val_RSMdataset_v2_seq30.lst