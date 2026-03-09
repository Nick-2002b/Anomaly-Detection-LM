import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from pathlib import Path
from pytorch_model_summary import summary


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

        # Strato convoluzionale, l'input ha 3 canali, l'output ne avra' 6.
        # Sara' analizzato da una "lente di ingrandimento" 5x5.
        self.conv1 = nn.Conv2d(3, 6, 5)

        # Strato di max pooling. Estrae il massimo valore concentrandosi,
        # posizione per posizione, su di una regione 2x2.
        self.pool1 = nn.MaxPool2d(2, 2)

        # Strato convoluzionale, l'input ha 6 canali, l'output ne avra' 16.
        # Sara' analizzato da una "lente di ingrandimento" 5x5.
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Strato di max pooling. Estrae il massimo valore concentrandosi,
        # posizione per posizione, su di una regione 2x2.
        self.pool2 = nn.MaxPool2d(2, 2)

        # Primo strato di neuroni completamente connessi.
        # In input la totalita' dei neuroni precedenti, in output 120.
        self.fc1 = nn.Linear(16 * 53 * 53, 120)

        # Secondo strato di neuroni completamente connessi.
        # In input 120 neuroni, in output 84.
        self.fc2 = nn.Linear(120, 84)

        # Secondo strato di neuroni completamente connessi.
        # In input 84 neuroni, in output un neurone per classe.
        self.fc3 = nn.Linear(84, len(classes))


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))    # Convoluzione, ReLU, MaxPooling.
        x = self.pool2(F.relu(self.conv2(x)))    # Convoluzione, ReLU, MaxPooling.
        x = torch.flatten(x, 1)                 # Tutti gli attuali neuroni vengono allineati in un unica 'fila'.
        x = F.relu(self.fc1(x))                 # Strato completamente connesso, ReLU.
        x = F.relu(self.fc2(x))                 # Strato completamente connesso, ReLU.
        x = self.fc3(x)                         # Strato completamente connesso.
        return x                                # Output.


    def get_transforms(self):
        return transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224,224)),
            transforms.ToDtype(torch.float32, scale=True)
        ])


    # Restituisce un tensore fittizzio da poter fare viaggiare nella rete.
    # - Utilizzato ad esempio per creare un summary di layer e parametri.
    # - Utilizzato ad esempio per il log della architettura su tensorboard.
    def get_dummy_input(self):

        # Definisco la tupla che rappresenta la dimensione dell'input che avra' la rete:
        # - batch size, canali, larghezza, altezza.
        input_shape = (1, 3, 224, 224)

        return torch.ones(size=input_shape, dtype=torch.float32).to(self.device)

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