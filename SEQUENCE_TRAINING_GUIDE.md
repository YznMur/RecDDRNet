# RecDDRNet Sequence Training Guide

## Summary of Changes

Your training dataset has been successfully converted from single-frame to non-overlapping sequences for training the ConvLSTM model.

### Conversion Results

```
Original frames: 43,012
Sequences (len=4): 10,720
Total frames used: 42,880
Frames skipped: 132 (incomplete sequences at video boundaries)

Videos processed: 193
```

### What Changed

1. **New list file**: `lists/train_RSMdataset_v2_seq4.lst`
   - Each line now contains 4 consecutive frames (non-overlapping)
   - Format: `img1 label1 img2 label2 img3 label3 img4 label4`
   - No overlapping sequences (prevents data leakage between train/val splits)

2. **Updated dataset loader**: `lib/datasets/rsm.py`
   - Now supports both single-frame and sequence formats
   - Automatically detects and loads pre-organized sequences
   - Maintains backward compatibility with original format

### How to Use

When training with prebuilt non-overlapping sequences, use the new list file in your config:

```yaml
DATASET:
  TRAIN_SET: "list/rsm/train_RSMdataset_v2_seq15.lst"
  # ... other settings

TRAIN:
  SEQUENCE_MODE: false
  SEQUENCE_LEN: 15
```

If you want to use a raw frame list and let the dataset build overlapping sequences, set `SEQUENCE_MODE: true` and `SEQUENCE_LEN` to the desired value. For non-overlapping training, generate a prebuilt sequence file instead.

### Generate a sequence list file for N frames

```bash
cd /home/yazan/RecDDRNet
python3 convert_to_sequences.py --input lists/train_RSMdataset_v2.lst --seq-len 15
```

This creates:

- `lists/train_RSMdataset_v2_seq15.lst`

### Generate train/val/test sequence lists together

```bash
cd /home/yazan/RecDDRNet
python3 convert_to_sequences.py --all --seq-len 15
```

This creates:

- `lists/train_RSMdataset_v2_seq15.lst`
- `lists/val_RSMdataset_v2_seq15.lst`
- `lists/test_RSMdataset_v2_seq15.lst`

Each line will contain 15 consecutive frames:

```text
img1 label1 img2 label2 ... img15 label15
```

### Network Processing

With the ConvLSTM model and sequence training:

```
Sequence of 4 frames
    ↓
[Frame 1] → DDRNet (extract features) → ConvLSTM (h₀, c₀) → Segmentation → Output 1
[Frame 2] → DDRNet (extract features) → ConvLSTM (h₁, c₁) → Segmentation → Output 2
[Frame 3] → DDRNet (extract features) → ConvLSTM (h₂, c₂) → Segmentation → Output 3
[Frame 4] → DDRNet (extract features) → ConvLSTM (h₃, c₃) → Segmentation → Output 4

Each frame is fully segmented, but with temporal memory from previous frames
```

### Key Benefits

- **No overlapping sequences**: Prevents data leakage during train/val splits
- **Temporal consistency**: ConvLSTM learns temporal patterns across 4 frames
- **Full segmentation**: Each frame produces its own segmentation output
- **Efficient training**: Fixed sequence size reduces data loading overhead

### Scripts Reference

- **Conversion script**: `convert_to_sequences.py`
  - Use if you need to regenerate sequences with different `SEQUENCE_LEN`
  - Automatically groups frames by video/job ID
  - Skips incomplete sequences to maintain uniform batch sizes

### Backing Up

- Original dataset: `lists/train_RSMdataset_v2.lst`
- Updated RSM loader: `lib/datasets/rsm.py`
- Backup of old loader: `lib/datasets/rsm_backup.py`
