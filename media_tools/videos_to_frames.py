import os
import cv2
import re
import subprocess
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# ------------------------------------------------------------------------------
# Convert milliseconds to HH-MM-SS-ms
# ------------------------------------------------------------------------------
def format_time(ms):
    seconds = int(ms // 1000)
    milliseconds = int(ms % 1000)

    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hrs:02d}-{mins:02d}-{secs:02d}-{milliseconds:03d}"


# ------------------------------------------------------------------------------
# Extract date from filename (DD-MM-YYYY)
# ------------------------------------------------------------------------------
def extract_date(filename):
    match = re.search(r'(\d{2}-\d{2}-\d{4})', filename)
    if not match:
        return datetime.min

    return datetime.strptime(match.group(1), "%d-%m-%Y")


# ------------------------------------------------------------------------------
# SCP transfer
# ------------------------------------------------------------------------------
def transfer_video(video_path):
    try:
        remote = "ymurhij@192.168.234.3:/home/ymurhij/video/"

        cmd = [
            "/usr/bin/scp",
            "-r",
            str(video_path),
            remote
        ]

        print(f"\nTransferring {video_path.name} via SCP ...")
        subprocess.run(cmd, check=True)
        print(f"Transfer complete: {video_path.name}")

    except subprocess.CalledProcessError as e:
        print(f"Transfer failed for {video_path.name}: {e}")


# ------------------------------------------------------------------------------
# Frame extraction + LST generation
# ------------------------------------------------------------------------------
def extract_frames(input_folder):
    input_folder = Path(input_folder)

    # ✅ HARD-CODED OUTPUT PATH
    output_folder = Path("/home/trainer/DDRNet.pytorch/data/rsm/test")

    list_dir = output_folder / "lists"
    list_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        input_folder.glob("*.*"),
        key=lambda x: extract_date(x.name)
    )

    print(f"Found {len(video_files)} videos")

    for video_path in tqdm(video_files, desc="Videos"):

        video_name = video_path.stem
        lst_path = list_dir / f"{video_name}.lst"

        # Skip if already processed
        if lst_path.exists():
            print(f"\nSkipping {video_name} (LST already exists)")
            continue

        print(f"\nProcessing: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open {video_path.name}")
            continue

        save_dir = output_folder / video_name
        save_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        saved = 0
        skipped = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with open(lst_path, "w") as lst_file:

            frame_bar = tqdm(
                total=total_frames,
                desc=f"{video_name}",
                leave=False
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                time_str = format_time(timestamp_ms)

                frame = cv2.resize(frame, (2048, 1024))

                filename = f"{video_name}_{frame_idx:08d}_{time_str}.jpg"
                save_path = save_dir / filename

                if save_path.exists():
                    skipped += 1
                else:
                    cv2.imwrite(str(save_path), frame)
                    saved += 1

                relative_path = f"test/{video_name}/{filename}"
                lst_file.write(relative_path + "\n")

                frame_idx += 1
                frame_bar.update(1)

            frame_bar.close()

        cap.release()

        print(f"Done: {video_name}")
        print(f"Saved frames: {saved}")
        print(f"Skipped existing: {skipped}")
        print(f"LST file: {lst_path}")
        print(f"Output folder: {save_dir}")

        # Transfer after processing
        transfer_video(video_path)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Video → Frames + LST files + tqdm + SCP transfer"
    )

    parser.add_argument("--input", required=True, help="Folder with videos")

    args = parser.parse_args()

    extract_frames(args.input)


# Example:
# python videos_to_frames.py --input /home/trainer/data/video