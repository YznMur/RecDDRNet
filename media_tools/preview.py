import os
import cv2

DDRNET_ROOT = "./RSM_v2_rsm_640x480_5_super_classes_weighted_CombinedLoss/rsm/ddrnet23_slim_640x480_5classes/previews/t"
DATASET_ROOT = "./data/rsm/leftImg8bit/test"

OUTPUT_PREVIEW = "preview_overlay.mp4"
OUTPUT_ORIGINAL = "preview_original.mp4"

FPS = 30
ALPHA = 0.6


def build_index(folder, suffix):
    index = {}
    for fname in os.listdir(folder):
        if fname.endswith(suffix):
            key = fname.replace(suffix, "")
            index[key] = os.path.join(folder, fname)
    return index


def get_image_pairs(dataset_root, pred_root):
    dataset_index = build_index(dataset_root, "_leftImg8bit.png")
    pred_index = build_index(pred_root, "_gtFine_labelIds.jpg")

    pairs = []

    for key in dataset_index:
        if key in pred_index:
            pairs.append((dataset_index[key], pred_index[key]))

    print(f"Matched pairs: {len(pairs)}")
    return sorted(pairs)


def create_videos(pairs):
    if not pairs:
        print("No matching files found.")
        return

    img0 = cv2.imread(pairs[0][0])
    h, w, _ = img0.shape

    # Writers
    preview_writer = cv2.VideoWriter(
        OUTPUT_PREVIEW,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w * 2, h)
    )

    original_writer = cv2.VideoWriter(
        OUTPUT_ORIGINAL,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h)
    )

    for img_path, pred_path in pairs:
        img = cv2.imread(img_path)
        pred = cv2.imread(pred_path)

        if img is None or pred is None:
            continue

        pred = cv2.resize(pred, (w, h))

        # overlay video
        overlay = cv2.addWeighted(img, 1 - ALPHA, pred, ALPHA, 0)
        combined = cv2.hconcat([img, overlay])
        preview_writer.write(combined)

        # original-only video
        original_writer.write(img)

    preview_writer.release()
    original_writer.release()

    print(f"Saved overlay video to {OUTPUT_PREVIEW}")
    print(f"Saved original video to {OUTPUT_ORIGINAL}")


if __name__ == "__main__":
    pairs = get_image_pairs(DATASET_ROOT, DDRNET_ROOT)
    create_videos(pairs)