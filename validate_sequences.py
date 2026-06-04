#!/usr/bin/env python3
"""
Validate that the sequence-based dataset loads correctly.
"""

import os
import sys

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from datasets.rsm import RSM

def validate_sequences():
    """Test loading sequences from the new list file."""
    
    root = '/home/yazan/RecDDRNet/data/rsm'
    list_path = 'list/rsm/train_RSMdataset_v2_seq4.lst'
    
    print("Initializing RSM dataset with sequences...")
    try:
        dataset = RSM(
            root=root,
            list_path=list_path,
            num_samples=None,
            num_classes=5,
            multi_scale=False,
            flip=False,
            sequence=False,  # Don't try to create sequences from sequences
            sequence_len=1
        )
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total sequences: {len(dataset)}")
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    # Try to load a few samples
    print("\nTesting sample loading...")
    test_indices = [0, len(dataset)//2, len(dataset)-1]
    
    for idx in test_indices:
        try:
            print(f"  Loading sample {idx}...", end=" ")
            image, label, size, name = dataset[idx]
            
            # Check output shapes
            if isinstance(image, list):
                num_frames = len(image)
                print(f"✓ Sequence of {num_frames} frames, size={size}, name={name}")
            else:
                print(f"✓ Shape={image.shape if hasattr(image, 'shape') else 'stacked'}, size={size}, name={name}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    print("\n✓ All tests passed! Dataset is ready for training.")
    return True

if __name__ == '__main__':
    success = validate_sequences()
    sys.exit(0 if success else 1)
