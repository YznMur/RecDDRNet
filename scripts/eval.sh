#!/bin/bash
python tools/eval.py --cfg configs/rsm_ddrnet23slim_convlstm.yaml \
    --splits test,train,val \
    DATASET.TRAIN_SET list/rsm/train_RSMdataset_v2_seq30.lst \
    DATASET.TEST_SET list/rsm/val_RSMdataset_v2_seq30.lst \
    DATASET.EVAL_TEST_SET list/rsm/test_RSMdataset_v2_seq30.lst
