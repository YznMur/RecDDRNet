from pathlib import Path

def create_test_lst_file():
    # --- Define paths (absolute for safety) ---
    left_img_base = Path("data/cityscapes/leftImg8bit/val").resolve()
    gt_fine_base = Path("data/cityscapes/gtFine/val").resolve()
    output_file = "val_RSMdataset_v2.lst"

    print(f"Using paths:\n  left_img_base={left_img_base}\n  gt_fine_base={gt_fine_base}")

    if not left_img_base.exists():
        print(f"❌ ERROR: Left image folder not found: {left_img_base}")
        return
    if not gt_fine_base.exists():
        print(f"❌ ERROR: Label folder not found: {gt_fine_base}")
        return

    # --- Collect images recursively (no need for subfolders) ---
    left_img_files = sorted(left_img_base.rglob("*.png"))
    print(f"✅ Found {len(left_img_files)} images in {left_img_base}")

    if not left_img_files:
        print("⚠️ No images found! Check your folder structure.")
        return

    missing_labels = 0

    with open(output_file, "w") as f:
        for i, left_img_file in enumerate(left_img_files):
            print(left_img_file)
            base_name = left_img_file.name.replace("_leftImg8bit.png", "")
            gt_fine_file = gt_fine_base / f"{base_name}_gtFine_labelIds.png"

            if gt_fine_file.exists():
                f.write(f"{left_img_file}\t{gt_fine_file}\n")
            else:
                f.write(f"{left_img_file}\n")
                missing_labels += 1
                print(f"⚠️ Missing label: {gt_fine_file.name}")

    total_written = sum(1 for _ in open(output_file))
    print(f"\n📄 Created {output_file} with {total_written} entries")
    if missing_labels > 0:
        print(f"⚠️ {missing_labels} images skipped because labels were missing.")

if __name__ == "__main__":
    create_test_lst_file()
