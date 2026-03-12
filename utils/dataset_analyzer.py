import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_mvtec_category(base_path: str):
    base_dir = Path(base_path).resolve()

    # 1. Analisi del Training Set
    train_good_dir = base_dir / "train" / "good"
    if not train_good_dir.exists():
        print(f"ERROR: Training folder not found in: {train_good_dir}")
        return

    train_images = list(train_good_dir.glob("*.png"))
    print(f"--- DATASET INFO: {base_dir.name.upper()} ---")
    print(f"Training images (only good one): {len(train_images)}")

    # 2. Analisi del Test Set
    test_dir = base_dir / "test"
    test_categories = [d for d in test_dir.iterdir() if d.is_dir()]

    print("\nTest images:")
    total_test = 0
    anomalous_categories = []

    for cat in test_categories:
        num_imgs = len(list(cat.glob("*.png")))
        print(f"  - {cat.name}: {num_imgs} images")
        total_test += num_imgs
        if cat.name != "good":
            anomalous_categories.append(cat.name)

    print(f"Total test images: {total_test}")

    # 3. Visualizzazione di un esempio
    # Prendiamo una prima immagine di training
    sample_good = cv2.imread(str(train_images[0]))
    sample_good = cv2.cvtColor(sample_good, cv2.COLOR_BGR2RGB)  # trasformazione da BGR in RGB

    # Prendiamo un'immagine di test anomala a caso (dalla prima categoria di difetti trovata)
    if anomalous_categories:
        defect_cat = anomalous_categories[0]
        sample_bad_path = list((test_dir / defect_cat).glob("*.png"))[0]

        sample_bad = cv2.imread(str(sample_bad_path))
        sample_bad = cv2.cvtColor(sample_bad, cv2.COLOR_BGR2RGB)

        # Recuperiamo la maschera (ground truth) corrispondente
        # Il nome del file maschera di solito finisce con _mask.png
        mask_name = sample_bad_path.stem + "_mask.png"
        mask_path = base_dir / "ground_truth" / defect_cat / mask_name

        if mask_path.exists():
            sample_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            sample_mask = None
            print(f"Warning: Ground truth mask not found for {mask_path}")

        # Plottiamo tutto
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(sample_good)
        axes[0].set_title("Training - Good")
        axes[0].axis("off")

        axes[1].imshow(sample_bad)
        axes[1].set_title(f"Test - Anomalous ({defect_cat})")
        axes[1].axis("off")

        if sample_mask is not None:
            axes[2].imshow(sample_mask, cmap='gray')
            axes[2].set_title("Ground Truth Mask")
            axes[2].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Risale alla root del progetto partendo dalla posizione di questo script
    dataset_path = Path(__file__).parent.parent / "data" / "bottle"
    analyze_mvtec_category(dataset_path)