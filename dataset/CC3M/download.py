import os
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# ===================== Configuration =====================
SAVE_ROOT = os.environ.get("STARR_CC3M_DIR", "data/CC3M/images")
# ==================================================

os.makedirs(f"{SAVE_ROOT}/train", exist_ok=True)
os.makedirs(f"{SAVE_ROOT}/validation", exist_ok=True)


def save_cc3m_images(split: str, save_dir: str):
    print(f"Loading the {split} dataset...")
    ds = load_dataset("pixparse/cc3m-wds", split=split, streaming=True)

    # Collect filenames already downloaded.
    existing = set(os.listdir(save_dir))
    print(f"Skipping {len(existing)} existing images...")

    saved = 0
    skipped = 0

    for sample in tqdm(ds, desc=f"{split}"):
        # Use the dataset key as a stable filename.
        key = sample["__key__"]
        img_name = f"{key}.jpg"

        if img_name in existing:
            skipped += 1
            continue

        try:
            img = sample["jpg"]
            img.convert("RGB").save(os.path.join(save_dir, img_name), "JPEG")
            existing.add(img_name)
            saved += 1
        except Exception:
            continue

    print(f"{split} complete: saved {saved}, skipped {skipped} images")


if __name__ == "__main__":
    save_cc3m_images("train", f"{SAVE_ROOT}/train")
    # save_cc3m_images("validation", f"{SAVE_ROOT}/validation")
