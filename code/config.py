import os

# ── Paths ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ── Data ───────────────────────────────────────────
NUM_IMAGES = 200
TRAIN_RATIO = 0.8
MAX_CAPTION_LENGTH = 32

# ── Vision Encoder ─────────────────────────────────
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"
VISION_HIDDEN_SIZE = 768          # CLIP ViT-B/32 output dim
VISION_NUM_PATCHES = 50           # 49 patches + CLS token
IMAGE_SIZE = 224

# ── Mini Q-Former ──────────────────────────────────
QF_NUM_QUERIES = 8
QF_HIDDEN_DIM = 256
QF_NUM_LAYERS = 2
QF_NUM_HEADS = 8
QF_FF_DIM = QF_HIDDEN_DIM * 4
QF_DROPOUT = 0.1

# ── Language Decoder ───────────────────────────────
LANGUAGE_MODEL_NAME = "facebook/opt-125m"

# ── Training ───────────────────────────────────────
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LOG_INTERVAL = 10          # log loss every N batches
VAL_EPOCH_INTERVAL = 5     # run validation every N epochs

# ── GPU / Mixed Precision ───────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True          # Automatic Mixed Precision: float16 where safe
NUM_WORKERS = 2         # DataLoader worker processes
PIN_MEMORY = True       # Pin memory for faster CPU→GPU transfer
