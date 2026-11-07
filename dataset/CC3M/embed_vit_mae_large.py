import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTMAEModel, ViTImageProcessor

# ===================== Configuration =====================
IMAGE_DIR = os.environ.get("STARR_IMAGE_DIR", "data/CC3M/images/train")
OUTPUT_FILE = os.environ.get("STARR_OUTPUT_FILE", "cc3m_vit_mae_large.npy")
CKPT_FILE = os.environ.get("STARR_CKPT_FILE", "cc3m_vit_mae_large_ckpt.npy")

MODEL_NAME = os.environ.get("STARR_MODEL_NAME", "facebook/vit-mae-large")
# Output dimension: 1024 (ViT-Large CLS token).
# ViTImageProcessor normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]

TOTAL = 1_000_000
BATCH_SIZE = 128  # Use a smaller batch for the 307M-parameter MAE-Large encoder.
NUM_WORKERS = 8
# ==================================================

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ---------- Load model ----------
print(f"Loading model: {MODEL_NAME}")
# ViTMAEModel loads the encoder; decoder weights are not used here.
model = ViTMAEModel.from_pretrained(MODEL_NAME)
# Use the image processor associated with this checkpoint.
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()


def preprocess(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


# ---------- Dataset ----------
class ImageFolderDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = preprocess(path)
        if img is None:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return img, idx


def collate_fn(batch):
    pil_images, indices = zip(*batch)
    inputs = processor(
        images=list(pil_images),
        return_tensors="pt",
    )
    return inputs["pixel_values"], list(indices)


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
if os.path.exists(CKPT_FILE):
    done_embs = np.load(CKPT_FILE)
    done_count = len(done_embs)
    all_embeddings = [done_embs]
    all_files = all_files[done_count:]
    print(f"Resuming: {done_count} images completed, {len(all_files)} remaining")
else:
    done_count = 0
    all_embeddings = []
    print("Starting from scratch")

# ---------- DataLoader ----------
dataset = ImageFolderDataset(all_files)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
    prefetch_factor=2,
)

# ---------- Inference ----------
batch_count = 0
with torch.no_grad():
    for pixel_values, indices in tqdm(loader, desc="Encoding"):
        pixel_values = pixel_values.to(device, non_blocking=True)

        # Use the model default masking behavior when noise is not supplied.
        # last_hidden_state shape: (B, num_patches+1, 1024)
        # Token 0 is the CLS token.
        outputs = model(pixel_values=pixel_values, noise=None)
        embs = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
        embs = F.normalize(embs, dim=-1)

        all_embeddings.append(embs.cpu().numpy().astype(np.float16))
        batch_count += 1

        if batch_count % 10 == 0:
            ckpt = np.concatenate(all_embeddings, axis=0)
            np.save(CKPT_FILE, ckpt)
            total_done = done_count + len(ckpt)
            print(f"\nCheckpoint saved: {total_done} images")

# ---------- Save final embeddings ----------
final = np.concatenate(all_embeddings, axis=0)
np.save(OUTPUT_FILE, final)
print(f"\nDone: shape={final.shape}, saved to {OUTPUT_FILE}")
# Expected shape: (TOTAL, model output dimension).

if os.path.exists(CKPT_FILE):
    os.remove(CKPT_FILE)
    print(f"Removed checkpoint: {CKPT_FILE}")
