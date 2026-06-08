#!/usr/bin/env python3
"""
Convert frame-by-frame list to non-overlapping sequences.
Each line in the output will contain SEQUENCE_LEN consecutive frames.
Format: img1 label1 img2 label2 img3 label3 ... imgN labelN
"""

import argparse
import os
import sys

DEFAULT_SEQUENCE_LEN = 4

SEQUENCE_LEN = DEFAULT_SEQUENCE_LEN

def group_by_video(lines):
    """Group frames by video/job."""
    videos = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        
        img_path = parts[0]
        label_path = parts[1]
        
        # Extract video ID from filename.
        # e.g. "job_1015__frame_000000_..." or "job_2685__0_..." -> "job_1015"
        img_name = os.path.basename(img_path)
        if '__frame_' in img_name:
            video_id = img_name.split('__frame_')[0]
        elif '__' in img_name:
            video_id = img_name.split('__')[0]
        else:
            video_id = os.path.splitext(img_name)[0]
        
        if video_id not in videos:
            videos[video_id] = []
        
        videos[video_id].append((img_path, label_path))
    
    return videos

def create_sequences(videos, seq_len=SEQUENCE_LEN):
    """Create non-overlapping sequences from grouped frames."""
    sequences = []
    
    for video_id, frames in sorted(videos.items()):
        # Create non-overlapping sequences (stride = seq_len)
        for i in range(0, len(frames), seq_len):
            seq = frames[i:i+seq_len]
            
            # Skip if sequence is incomplete (not enough frames)
            if len(seq) < seq_len:
                print(f"  Skipping incomplete sequence from {video_id} at frame {i}: {len(seq)} frames (need {seq_len})")
                continue
            
            # Flatten sequence: img1 label1 img2 label2 ...
            seq_line = ' '.join([f"{img} {label}" for img, label in seq])
            sequences.append(seq_line)
    
    return sequences

def write_sequences(output_file, sequences):
    print(f"\nWriting to: {output_file}")
    with open(output_file, 'w') as f:
        for seq in sequences:
            f.write(seq + '\n')

    print("✓ Conversion complete!")


def process_file(input_file, output_file, sequence_len):
    if sequence_len <= 1:
        raise ValueError('Sequence length must be greater than 1 for sequence conversion.')

    print(f"Reading frames from: {input_file}")
    with open(input_file, 'r') as f:
        lines = f.readlines()

    print(f"Total frames: {len(lines)}")

    print("\nGrouping frames by video...")
    videos = group_by_video(lines)
    print(f"Found {len(videos)} videos:")
    for vid, frames in sorted(videos.items()):
        print(f"  {vid}: {len(frames)} frames")

    print(f"\nCreating non-overlapping sequences (SEQUENCE_LEN={sequence_len})...")
    sequences = create_sequences(videos, seq_len=sequence_len)

    print(f"Total sequences created: {len(sequences)}")
    print(f"  Total frames used: {len(sequences) * sequence_len}")
    print(f"  Frames skipped: {len(lines) - len(sequences) * sequence_len}")

    write_sequences(output_file, sequences)


def main():
    parser = argparse.ArgumentParser(
        description='Convert frame-by-frame lists into non-overlapping sequence lists.'
    )
    parser.add_argument('--input', default=None,
                        help='Input frame list file (two columns per line: img label)')
    parser.add_argument('--seq-len', type=int, default=DEFAULT_SEQUENCE_LEN,
                        help='Number of frames per sequence')
    parser.add_argument('--output', default=None,
                        help='Output sequence list file')
    parser.add_argument('--all', action='store_true',
                        help='Process train, val, and test lists automatically')
    parser.add_argument('--dir', default='data/list/rsm',
                        help='Directory containing train/val/test list files')

    args = parser.parse_args()
    sequence_len = args.seq_len

    if args.all:
        file_names = [
            'train_RSMdataset_v2.lst',
            'val_RSMdataset_v2.lst',
            'test_RSMdataset_v2.lst',
        ]
        for file_name in file_names:
            input_file = os.path.join(args.dir, file_name)
            output_file = os.path.join(args.dir, file_name.replace('.lst', f'_seq{sequence_len}.lst'))
            process_file(input_file, output_file, sequence_len)
    else:
        if not args.input:
            raise ValueError('Either --input or --all must be provided.')
        input_file = args.input
        output_file = args.output or input_file.replace('.lst', f'_seq{sequence_len}.lst')
        process_file(input_file, output_file, sequence_len)
if __name__ == '__main__':
    main()
