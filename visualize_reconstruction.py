import torch
from pathlib import Path
from nets.simple_autoencoder import SimpleAutoencoder
from utils.visual_util import ColoredPrint as cp

def visualize_reconstruction():
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR / "runs" / "base_autoencoder.pth"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = SimpleAutoencoder.to(device)

    if not MODEL_PATH.exists():
        cp.red(f"Model not found: {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()