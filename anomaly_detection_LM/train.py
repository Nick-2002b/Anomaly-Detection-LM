import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from nets.simple_autoencoder import SimpleAutoencoder
from pathlib import Path
from utils.visual_util import ColoredPrint as cp
from utils.mvtec_dataset import MVTecDataset
import math

def train_baseline():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_ROOT = BASE_DIR / "data"
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 10

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cp.green(f"Device: {device}")

    transform_pipeline = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    train_dataset = MVTecDataset(root_dir=DATA_ROOT, category="bottle", is_train=True, transform=transform_pipeline)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True, # Velocizza il trasferimento dati verso la GPU
        num_workers=8,
        prefetch_factor=2 # La cpu prepara 2 batch in anticipi da passare alla GPU
    )
    model = SimpleAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    tr_steps_per_epoch = math.ceil(len(train_dataset)/BATCH_SIZE)

    cp.purple(f"Training starts for {EPOCHS} epoche...")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for epoch_step, (images, _, _) in enumerate(train_loader):

            cp.cyan(f'Epoch: {epoch + 1}, Step: {epoch_step + 1}/{tr_steps_per_epoch}')

            # Spostiamo le immagini sulla scheda video (non-blocking aiuta con pin_memory)
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad() # azzerra i gradienti(errori nei pesi)

            # Forward pass: l'immagine attraversa l'autoencoder
            reconstructed = model(images)

            # Calcolo della Loss: confrontiamo l'ouput con l'Input
            loss = criterion(reconstructed, images)

            loss.backward()

            # aggiorna i pesi della rete
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        cp.purple(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.6f}")

    cp.green(f"Training ends for {EPOCHS} epoche...")

    save_dir = BASE_DIR/"runs"
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir/"base_autoencoder.pth")
    cp.yellow(f"Model saved in {save_dir}/base_autoencoder.pth")

if __name__ == "__main__":
    train_baseline()


