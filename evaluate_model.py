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

    y_true = [] # Le etichette reali (0 = buone, 1 = anomalia)
    y_scores = [] # Il punteggio di anomalia predetto

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)

            reconstructed = model(images)

            # Calcoliamo l'errore quadratico (MSE) per ogni pixel nella singola immagine
            mse_map = (images - reconstructed) ** 2

            # Calcoliamo l'Anomaly Score: prendiamo la media degli errori per canale,
            # e poi il valore MASSIMO di tutta la mappa dell'immagine.
            # Se c'è un'anomalia, ci sarà almeno un pixel con un errore altissimo.
            mean_channels = torch.mean(mse_map, dim=1)
            anomaly_score = torch.max(mean_channels).item()

            y_true.append(labels.item())
            y_scores.append(anomaly_score)

    auroc = roc_auc_score(y_true, y_scores)

    cp.yellow("\n--- EVALUATION RESULT ---")
    cp.cyan(f"Dataset> {CATEGORY.upper()}")
    cp.cyan(f"AUROC Score: {auroc:.4f} ({auroc * 100:.2f}%)")

    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)

    cp.cyan(f"Average score error to reconstruct GOOD images: {np.mean(y_scores_np[y_true_np == 0]):.4f}")
    cp.cyan(f"Average score error to reconstruct ANOMALOUS images: {np.mean(y_scores_np[y_true_np == 1]):.4f}")

if __name__ == "__main__":
    evaluate_baseline()