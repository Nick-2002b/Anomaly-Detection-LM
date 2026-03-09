import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from pathlib import Path
from pytorch_model_summary import summary

from torchvision import models
from torchvision.models.vgg import VGG11_Weights


# Rete di classificazione immagini.
class Net(nn.Module):


    def __init__(self, classes : list[str]) -> None:
        super(Net, self).__init__()
        
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Carico VGG16 pre-addestrato
        self.pre_trained_model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)

        # Congelo tutti i pesi della rete (feature extraction)
        for param in self.pre_trained_model.features.parameters():
            param.requires_grad = False

        # Sostituisco il classificatore finale (1000 -> numero classi)
        self.pre_trained_model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=len(classes))


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.pre_trained_model(x)
        return x


    def get_transforms(self):
        return transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224,224)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    # Restituisce un tensore fittizzio da poter fare viaggiare nella rete.
    # - Utilizzato ad esempio per creare un summary di layer e parametri.
    # - Utilizzato ad esempio per il log della architettura su tensorboard.
    def get_dummy_input(self):

        # Definisco la tupla che rappresenta la dimensione dell'input che avra' la rete:
        # - batch size, canali, larghezza, altezza.
        input_shape = (1, 3, 224, 224)

        return torch.ones(size=input_shape).to(self.device)


if __name__ == '__main__':

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    out_root = Path('./out')

    # Il percorso indicato esiste?
    if not out_root.exists():
        out_root.mkdir()

    # Crea l'oggetto che rappresenta la rete.
    # Fornisce le classi.
    n = Net(['a', 'b', 'c']).to(device)

    # Salva i parametri addestrati della rete.
    torch.save(n.state_dict(), './out/model_state_dict.pth')

    # Salva l'intero modello.
    torch.save(n, './out/model.pth')

    # Stampa informazioni generali sul modello.
    print(n)

    # Stampa i parametri addestrabili.
    # for name, param in n.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    # Stampa un recap del modello.
    print(summary(n, n.get_dummy_input()))