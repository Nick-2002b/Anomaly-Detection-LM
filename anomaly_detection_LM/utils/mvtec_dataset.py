from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MVTecDataset(Dataset):
    def __init__(self, root_dir: str, category: str, is_train: bool = True, transform=None):
        """
        Dataset custom for MVTec AD.
        """
        # Creiamo il percorso base, es: ./data/bottle
        self.root_dir = Path(root_dir) / category
        self.is_train = is_train
        self.transform = transform

        self.image_paths = []
        self.labels = []  # 0 = immagine normale, 1 = immagine con anomalia

        if self.is_train:
            # In training carichiamo SOLO le immagini perfette
            good_dir = self.root_dir / "train" / "good"

            if not good_dir.exists():
                raise FileNotFoundError(f"Training folder not found: {good_dir}")

            for img_path in good_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(0)
        else:
            test_dir = self.root_dir / "test"
            if not test_dir.exists():
                raise FileNotFoundError(f"Test folder not found: {test_dir}")

            for defect_dir in test_dir.iterdir():
                if defect_dir.is_dir():
                    label = 0 if defect_dir.name == "good" else 1
                    for img_path in defect_dir.glob("*.png"):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Usiamo PIL per caricare l'immagine perché torchvision.transforms lavora meglio con PIL
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.resolve()
    DATA_ROOT = BASE_DIR / "data"
    CATEGORY = "bottle"

    # Definiamo le trasformazioni: ridimensioniamo e convertiamo in Tensore PyTorch
    transform_pipeline = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    print("Initializing Training Dataset")
    train_dataset = MVTecDataset(DATA_ROOT, CATEGORY, is_train=True, transform=transform_pipeline)
    print(f"Found {len(train_dataset)} images in training set.")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    images, labels, paths = next(iter(train_loader))

    print("\nBatch extraction test completed:")
    print(f"Image batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
