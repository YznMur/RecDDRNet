#!/bin/bash
# Evaluate train/val/test with seq20 lists - single model load

CFG="configs/rsm_ddrnet23slim_convlstm.yaml"

python3 tools/eval.py --cfg $CFG --splits all \
    DATASET.TRAIN_SET list/rsm/train_RSMdataset_v2_seq20.lst \
    DATASET.TEST_SET list/rsm/val_RSMdataset_v2_seq20.lst \
    DATASET.EVAL_TEST_SET list/rsm/test_RSMdataset_v2_seq20.lst
