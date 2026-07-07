#!/bin/bash
python tools/eval.py --cfg configs/rsm_ddrnet23slim_keyframe_updater_attn.yaml \
    --splits test,train,val \
    DATASET.TRAIN_SET list/rsm/train_RSMdataset_v2_seq20.lst \
    DATASET.TEST_SET list/rsm/val_RSMdataset_v2_seq20.lst \
    DATASET.EVAL_TEST_SET list/rsm/test_RSMdataset_v2_seq20.lst
