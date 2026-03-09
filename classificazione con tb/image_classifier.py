import time
import shutil

from pathlib import Path
from net_runner import NetRunner
from visual_util import ColoredPrint as cp
from torch.utils.tensorboard import SummaryWriter
from config_helper import check_and_get_configuration
from custom_dataset_shapes import CustomDatasetShapes
import torch

# Ottengo il percorso assoluto della cartella in cui si trova questo script
BASE_DIR = Path(__file__).parent.resolve()
CONFIG = str(BASE_DIR / 'config' / 'config.json')
SCHEMA = str(BASE_DIR / 'config' / 'config_schema.json')

TB_DIR = './runs'


def print_dataset_statistics(cfg_obj):
    """Stampa il numero di immagini nei dataset di training, validation e test."""
    datasets = {
        "Training": CustomDatasetShapes(root=cfg_obj.io.training_folder, transform=None),
        "Validation": CustomDatasetShapes(root=cfg_obj.io.validation_folder, transform=None),
        "Test": CustomDatasetShapes(root=cfg_obj.io.test_folder, transform=None),
    }

    for name, dataset in datasets.items():
        print(f"{name} dataset: {len(dataset)} immagini")


if __name__ == "__main__":

    # Carica il file di configurazione, lo valido e ne creo un oggetto a partire dal json.
    cfg_obj = check_and_get_configuration(CONFIG, SCHEMA)

    # Uso un data loader semplicemente per ricavare le classi del dataset.
    classes = CustomDatasetShapes(root=cfg_obj.io.training_folder, transform=None).classes
    
    print_dataset_statistics(cfg_obj)

    # Cancella, se presente, la vecchia cartella di log della tensorboard.
    if Path(TB_DIR).exists():
        cp.yellow(f'Cleaning old runs folder: {TB_DIR}')
        shutil.rmtree(TB_DIR)

        # Do tempo al sistema di cancellare la cartella.
        time.sleep(5)

    # Inizializza il logger della tensorboard.
    cp.yellow('Creating summary writer for tensorboard logs...')
    writer = SummaryWriter(TB_DIR)

    # Creo l'oggetto che mi permettera' di addestrare e testare il modello.
    runner = NetRunner(cfg_obj, classes, writer)

    # Se richiesto, eseguo il training.
    if cfg_obj.parameters.train:
        cp.purple(f"Test accuracy with initial model: {runner.test(use_current_net = True) * 100:.1f}%")
        runner.train()

    # Se richiesto, eseguo il test.
    if cfg_obj.parameters.test:
        cp.purple(f"Test accuracy with best model   : {runner.test(use_current_net = False) * 100:.1f}%")

    # Chiude il logger della tensorboard.
    writer.flush()
    writer.close()