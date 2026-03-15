import torch
from pathlib import Path

from cv2.datasets import none

from nets.simple_autoencoder import SimpleAutoencoder
from utils.visual_util import ColoredPrint as cp
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.mvtec_dataset import MVTecDataset

def visualize_reconstruction():
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR / "runs" / "base_autoencoder.pth"
    DATA_ROOT = BASE_DIR / "data"
    CATEGORY = "bottle"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = SimpleAutoencoder.to(device)

    if not MODEL_PATH.exists():
        cp.red(f"Model not found: {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    transform_pipelne = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    test_dataset = MVTecDataset(DATA_ROOT, CATEGORY, is_train=False, transform=transform_pipelne)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True) # shuffle true cosi ogni volta vediamo un'immagine diversa

    cp.cyan(f"Finding an anomalous {CATEGORY} in the test set... ")

    original_img = None

    for img, label, path in test_loader:
        if label.item() == 1:
            original_img = img.to(device)
            img_path = path[0]
            break

    if original_img is None:
        cp.red(f"None image with anomalous found")
        return

    # Passiamo l'immagine nella rete senza calcolare i gradienti(risparmia RAM e tempo)
    with torch.no_grad():
        reconstructed_img = model(original_img)

    diff_map = torch.abs(original_img - reconstructed_img)

    # Trasforma la differenza tra l'immagine ricostruita e originale in una mappa in bianco e nero, per vedere le differenze
    anomaly_map = torch.mean(diff_map, dim=1).squeeze().cpu().numpy()

    # Converte le immagini per matplotlib (da [C, H, W] a [H, W, C])
    original_np = original_img.squeeze().cpu().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed_img.squeeze().cpu().permute(1, 2, 0).numpy()



