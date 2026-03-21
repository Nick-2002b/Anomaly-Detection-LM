import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pathlib import Path


from utils.visual_util import ColoredPrint as cp
from utils.mvtec_dataset import MVTecDataset
from nets.simple_autoencoder import SimpleAutoencoder

def evaluate_baseline():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_ROOT = BASE_DIR / "data"
    MODEL_PATH = BASE_DIR / "runs" / "base_autoencoder.pth"
    CATEGORY = "bottle"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = SimpleAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    transform_pipeline = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    test_dataset = MVTecDataset(DATA_ROOT, CATEGORY, is_train=False, transform=transform_pipeline)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    cp.yellow(f"Evaluating of {len(test_dataset)} test images...")

    y_true = []
    y_scores = []
