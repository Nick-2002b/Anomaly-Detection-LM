from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MVTecDataset(Dataset):
    def __init__(self, root_dir: str, category: str, is_train: bool = True, transform=None):
        """
        Dataset custom per MVTec AD.
        """
        # Creiamo il percorso base, es: ./data/bottle
        self.root_dir = Path(root_dir) / category
        self.is_train = is_train
        self.transform = transform

        self.image_paths = []
        self.labels = []  # 0 = immagine normale, 1 = immagine con anomalia

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass