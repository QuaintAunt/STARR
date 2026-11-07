import numpy as np
from tqdm import tqdm
import torch
import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset, DataLoader

# ===================== Configuration =====================
IMAGE_DIR = os.environ.get("STARR_IMAGE_DIR", "data/CC3M/images/train")
OUTPUT_FILE = os.environ.get("STARR_OUTPUT_FILE", "cc3m_dinov2-base.npy")
CKPT_FILE = os.environ.get("STARR_CKPT_FILE", "cc3m_dinov2_ckpt.npy")
CKPT_IDX_FILE = os.environ.get(
    "STARR_CKPT_IDX_FILE", "cc3m_dinov2_ckpt_idx.npy"
)  # Checkpoint progress metadata.
MODEL_NAME = os.environ.get("STARR_MODEL_NAME", "facebook/dinov2-base")
TOTAL = 1_000_000
BATCH_SIZE = 256
NUM_WORKERS = 8
# ==================================================

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            # squeeze: (1, C, H, W) -> (C, H, W)
            return inputs["pixel_values"].squeeze(0), idx
        except Exception:
            # Use a zero tensor to preserve the row for unreadable images.
            dummy = torch.zeros(3, 224, 224)
            return dummy, idx


def collate_fn(batch):
    pixels, indices = zip(*batch)
    return torch.stack(pixels), list(indices)


# ---------- Collect image paths ----------
all_files = sorted(
    [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)[:TOTAL]

print(f"Found {len(all_files)} images")

# ---------- Resume from checkpoint ----------
if os.path.exists(CKPT_FILE) and os.path.exists(CKPT_IDX_FILE):
    done_embs = np.load(CKPT_FILE)
    done_count = len(done_embs)
    all_embeddings = [done_embs]
    all_files = all_files[done_count:]  # Skip rows already encoded.
    print(f"Resuming: {done_count} images completed, {len(all_files)} remaining")
else:
    done_count = 0
    all_embeddings = []
    print("Starting from scratch")

# ---------- Create DataLoader ----------
dataset = ImageFolderDataset(all_files, processor)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
    prefetch_factor=2,
)

# ---------- Encoding ----------
batch_count = 0
with torch.no_grad():
    for pixels, indices in tqdm(loader, desc="Encoding"):
        pixels = pixels.to(device, non_blocking=True)
        outputs = model(pixel_values=pixels)
        # Use the CLS token as the image representation.
        embs = outputs.last_hidden_state[:, 0, :]  # (B, D)
        embs = torch.nn.functional.normalize(embs, dim=-1)  # L2 normalization
        all_embeddings.append(embs.cpu().numpy().astype(np.float16))
        batch_count += 1

        # Save a checkpoint every 10 batches.
        if batch_count % 10 == 0:
            ckpt = np.concatenate(all_embeddings, axis=0)
            np.save(CKPT_FILE, ckpt)
            np.save(CKPT_IDX_FILE, np.array([done_count + len(ckpt)]))
            print(f"\nCheckpoint saved: {done_count + len(ckpt)} images")

# ---------- Save final embeddings ----------
final = np.concatenate(all_embeddings, axis=0)

# The accumulated embeddings include the previously loaded checkpoint.
if done_count > 0:
    prev = np.load(CKPT_FILE)
    # The latest checkpoint may already contain batches from this run.
    # New batches are appended to all_embeddings after resuming.

np.save(OUTPUT_FILE, final)
print(f"\nDone: shape={final.shape}, saved to {OUTPUT_FILE}")

# Remove checkpoint files
for f in [CKPT_FILE, CKPT_IDX_FILE]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed checkpoint: {f}")
