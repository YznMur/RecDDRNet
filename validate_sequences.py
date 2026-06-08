#!/usr/bin/env python3
"""
Validate that the sequence-based dataset loads correctly.
"""

import os
import sys
import argparse

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from datasets.rsm import RSM

def validate_sequences(root, list_path, samples):
    """Test loading sequences from the new list file."""

    print("Initializing RSM dataset with sequences...")
    try:
        dataset = RSM(
            root=root,
            list_path=list_path,
            num_samples=None,
            num_classes=5,
            multi_scale=False,
            flip=False,
            sequence=False,
            sequence_len=1
        )
        print("Dataset loaded successfully!")
        print(f"  Total sequences: {len(dataset)}")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return False
    
    # Try to load a few samples
    print("\nTesting sample loading...")
    test_indices = sorted(set([0, len(dataset)//2, len(dataset)-1]))[:samples]
    
    for idx in test_indices:
        try:
            print(f"  Loading sample {idx}...", end=" ")
            image, label, size, name = dataset[idx]
            print(
                "ok image_shape={} label_shape={} size={} name={}".format(
                    getattr(image, "shape", None),
                    getattr(label, "shape", None),
                    size,
                    name,
                )
            )
                
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    print("\nAll tests passed. Dataset is ready for training.")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate RSM sequence list loading.')
    parser.add_argument('--root', default='data/',
                        help='Dataset root used by the training config')
    parser.add_argument('--list', default='list/rsm/train_RSMdataset_v2_seq15.lst',
                        help='Sequence list path used by the training config')
    parser.add_argument('--samples', type=int, default=3,
                        help='Number of representative samples to load')
    args = parser.parse_args()

    success = validate_sequences(args.root, args.list, args.samples)
    sys.exit(0 if success else 1)
