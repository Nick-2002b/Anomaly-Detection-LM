import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pathlib import Path
from visual_util import ColoredPrint as cp
from mvtec_dataset import MVTecDataset

def train_baseline():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_ROOT = BASE_DIR / "data"
    BATCH_SIZE = 16

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cp.green(f"Device: {device}")

    transform_pipeline = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    train_dataset = MVTecDataset(root_dir=DATA_ROOT, category="bottle", is_train=True, transform=transform_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
