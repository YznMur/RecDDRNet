#!/bin/bash
# Multi-GPU training with DDP (NCCL fixes for rsm4)
NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=^docker0,lo,br-* NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 MASTER_PORT=29501 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 tools/train.py --cfg configs/rsm_ddrnet23slim_convlstm.yaml