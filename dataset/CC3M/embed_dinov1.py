import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ===================== Configuration =====================
IMAGE_DIR = os.environ.get("STARR_IMAGE_DIR", "data/CC3M/images/train")
OUTPUT_FILE = os.environ.get("STARR_OUTPUT_FILE", "cc3m_dinov1_resnet50.npy")
CKPT_FILE = os.environ.get("STARR_CKPT_FILE", "cc3m_dinov1_resnet50_ckpt.npy")

DINO_REPO_DIR = str(
    __import__("pathlib").Path(__file__).resolve().parents[2] / "src/third_party/dino"
)
DINO_CKPT_PATH = os.environ.get("STARR_DINO_CKPT_PATH", "./dino_resnet50_pretrain.pth")

# Available models:
#   dino_vits16 / dino_vits8  → dimension 384
#   dino_vitb16 / dino_vitb8  → dimension 768
#   dino_xcit_small_12_p16 / dino_xcit_small_12_p8  → dimension 384
#   dino_xcit_medium_24_p16 / dino_xcit_medium_24_p8 → dimension 512
#   dino_resnet50             → dimension 2048
MODEL_NAME = os.environ.get("STARR_MODEL_NAME", "dino_resnet50")

TOTAL = 1_000_000
BATCH_SIZE = 256
NUM_WORKERS = 8
# ==================================================

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ---------- Load model ----------
print(f"Loading model: {MODEL_NAME}")
# model = torch.hub.load("facebookresearch/dino:main", MODEL_NAME)
model = torch.hub.load(
    DINO_REPO_DIR,  # Load the bundled DINO implementation.
    MODEL_NAME,
    source="local",  # Use the local repository.
    pretrained=not os.path.exists(
        DINO_CKPT_PATH
    ),  # Download pretrained weights only when no local checkpoint exists.
)

# Load the local weights when available.
if os.path.exists(DINO_CKPT_PATH):
    state_dict = torch.load(DINO_CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

# ---------- Preprocessing ----------
# Apply the ImageNet normalization used by DINOv1.
transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ---------- Dataset ----------
class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), idx
        except Exception:
            # Use a zero tensor to preserve the row for unreadable images.
            return torch.zeros(3, 224, 224), idx


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
dataset = ImageFolderDataset(all_files, transform)
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
    for pixels, indices in tqdm(loader, desc="Encoding"):
        pixels = pixels.to(device, non_blocking=True)

        # The encoder returns one feature vector per image: (B, D).
        embs = model(pixels)
        embs = F.normalize(embs, dim=-1)  # L2 normalization

        all_embeddings.append(embs.cpu().numpy().astype(np.float16))
        batch_count += 1

        # Save a checkpoint every 10 batches.
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

# Remove checkpoint files
if os.path.exists(CKPT_FILE):
    os.remove(CKPT_FILE)
    print(f"Removed checkpoint: {CKPT_FILE}")
