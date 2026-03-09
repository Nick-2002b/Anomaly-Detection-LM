import sys
import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from metrics import Metrics
from visual_util import ColoredPrint as cp
from custom_dataset_shapes import CustomDatasetShapes

torch.manual_seed(42)
np.random.seed(42)


class NetRunner():

    def __init__(self, cfg_object: object, classes: list[str], tb_writer) -> None:

        cp.purple('Initializing net runner...')

        # Memorizzo il logger della tensorboard.
        self.writer = tb_writer

        # Salvo il file di configurazione.
        self.cfg = cfg_object

        # Salvo le classi del dataset.
        self.classes = classes
        cp.cyan(f'Classifier classes: {self.classes}')

        # Predispone la cartella di output.
        self.out_root = Path(self.cfg.io.out_folder)

        # Il percorso indicato esiste?
        if not self.out_root.exists():
            cp.cyan(f'Creating output directory: {self.out_root}')
            self.out_root.mkdir()

        # Indico dove salvero' il modello addestrato.
        self.last_model_outpath_sd = self.out_root / 'last_model_sd.pth'
        self.last_model_outpath = self.out_root / 'last_model.pth'
        self.best_model_outpath_sd = self.out_root / 'best_model_sd.pth'
        self.best_model_outpath = self.out_root / 'best_model.pth'

        # Acquisisco la rete, in base al tipo richiesto.
        # - eventualmente si ricaricano i pesi dell'ultimo modello salvato su disco.
        
        cp.red("Choosing Device Type")
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        cp.green(f"{self.device} is the best for this Computer")
        
        self.net = self.__get_net('last' if self.cfg.train_parameters.reload_last_model else 'curr').to(self.device)

        # Registro l'architettura di rete nella tensorboard.
        self.writer.add_graph(self.net, self.net.get_dummy_input())

        # Carico e predispongo i loader dei dataset.
        self.__load_data()

        # Inizializzo i parametri utili all'addestramento.
        self.batch_size = cfg_object.hyper_parameters.batch_size
        self.lr = cfg_object.hyper_parameters.learning_rate # tasso di apprendimento.
        self.momentum = cfg_object.hyper_parameters.momentum # momentum.
        self.epochs = cfg_object.hyper_parameters.epochs # Numero di epoche di addestramento.

        # Tiene traccia degli step globali di addestramento.
        self.__global_step = 0

        self.last_step = self.epochs * self.tr_steps_per_epoch
        cp.cyan(f'Last step will be: {self.last_step}')

        # Funzione di costo.
        cp.cyan(f'Created loss function.')
        self.criterion = nn.CrossEntropyLoss()

        # Ottimizzatore.
        cp.cyan(f'Created optimizer (lr: {self.cfg.hyper_parameters.learning_rate}, m: {self.cfg.hyper_parameters.momentum}).')
        self.optimizer = optim.SGD(self.net.parameters(),
                                   lr = self.lr,
                                   momentum = self.momentum)

    def __load_data(self) -> None:

        transforms = self.net.get_transforms()

        # Posso raccogliere le immagini con il dataset custom creato appositamente.
        # - Posso farlo sia per i dati di training.
        # - Che per quelli di test e/o validazione.
        cp.cyan(f'Analyzing training dataset: {self.cfg.io.training_folder}')
        tr_dataset = CustomDatasetShapes(root=self.cfg.io.training_folder, transform=transforms)
        cp.cyan(f'Analyzing validation dataset: {self.cfg.io.validation_folder}')
        va_dataset = CustomDatasetShapes(root=self.cfg.io.validation_folder, transform=transforms)
        cp.cyan(f'Analyzing test dataset: {self.cfg.io.test_folder}')
        te_dataset = CustomDatasetShapes(root=self.cfg.io.test_folder, transform=transforms)
        self.classes = tr_dataset.classes

        # Se non voglio usare il dataset custom, posso usarne uno di base fornito da PyTorch.
        # Questo rappresenta genericamente:
        # - Un dataset di immagini.
        # - Diviso in sotto-cartelle.
        # - Il nome delle sotto-cartelle rappresenta il nome della classe.
        # - In ogni sotto-cartella ci sono solo immagini di quella classe.
        if not self.cfg.io.use_custom_dataset:
            tr_dataset = torchvision.datasets.ImageFolder(root=self.cfg.io.training_folder, transform=transforms)
            va_dataset = torchvision.datasets.ImageFolder(root=self.cfg.io.validation_folder, transform=transforms)
            te_dataset = torchvision.datasets.ImageFolder(root=self.cfg.io.test_folder, transform=transforms)

        # Creo poi i dataloader che prendono i dati dal dataset:
        # - lo fanno a pezzi di dimensione 'use_custom_dataset'.
        # - i pezzi li compongono di campioni rando se abilitato 'shuffle'.
        self.tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=self.cfg.hyper_parameters.batch_size, shuffle=True)
        self.va_loader = torch.utils.data.DataLoader(va_dataset, batch_size=self.cfg.hyper_parameters.batch_size, shuffle=False)
        self.te_loader = torch.utils.data.DataLoader(te_dataset, batch_size=self.cfg.hyper_parameters.batch_size, shuffle=False)

        self.tr_steps_per_epoch = math.ceil(len(tr_dataset) / self.tr_loader.batch_size)
        cp.cyan(f'Training steps per epoch will be: {self.tr_steps_per_epoch}')


    def train(self) -> None:

        cp.purple("Training...")

        # Reset degli step globali.
        self.__global_step = 0

        cp.cyan(f'Training loop epochs: {self.epochs}')

        # Salvo in una variabile in modo da mostrare una sola volta.
        preview = self.cfg.parameters.show_preview

        # Ogni quanti step stampare a video informazioni sullo stato dell'addestramento.
        console_log_step_cadence = self.cfg.train_parameters.console_log_steps
        # Ogni quanti step calcolare le metriche di addestramento: loss... e loggarle su tb.
        compute_metrics_step_cadence = self.cfg.train_parameters.metrics_computation_steps
        # Ogni quanti step verificare il raggiuntimento del target richiesto.
        target_reached_check_step_cadence = self.cfg.train_parameters.target_evaluation_steps
        # Il target di accuracy da raggiungere.
        self.__target = self.cfg.train_parameters.target

        cp.cyan(f'Training console log every {console_log_step_cadence} steps, preview: {preview}.')
        cp.cyan(f'Training metrics and tensorboard log every {compute_metrics_step_cadence} steps.')
        cp.cyan(f'Training and validation target evaluated every {target_reached_check_step_cadence} steps.')
        cp.cyan(f'Target to reach is {self.__target}% accuracy.')

        es_start_at_step = self.cfg.early_stop_parameters.es_start_at_step
        early_stop_check_step_cadence = self.cfg.early_stop_parameters.es_check_steps
        self.__es_patience = self.cfg.early_stop_parameters.patience
        self.es_improvement_rate = self.cfg.early_stop_parameters.improvement_rate

        cp.cyan(f'Early stop check will start at step {es_start_at_step}.')
        cp.cyan(f'Validation loss evaluated every {early_stop_check_step_cadence} steps.')
        cp.cyan(f'Early stop will be triggered after {self.__es_patience} times of no improvement.')
        cp.cyan(f'Minimum requested improvement is {self.es_improvement_rate}% on validation loss.')

        self.__best_tr_acc  = float('-inf') # Traccia la migliore accuracy raggiunta in addestramento.
        self.__best_va_acc  = float('-inf') # Traccia la migliore accuracy raggiunta in validazione.
        self.__best_tr_loss = float('inf')  # Traccia la migliore loss raggiunta in training.
        self.__best_va_loss = float('inf')  # Traccia la migliore loss raggiunta in validazione.

        self.__curr_tr_loss = None
        self.__curr_va_loss = None
        self.__curr_tr_acc  = None
        self.__curr_va_acc  = None

        # Con questo contatore, si valuta per quanti check consecutivi
        # la loss di validazione non e' migliorata.
        self.__patience_counter = 0

        target_reached = False      # FLAG EVENTO: il training puo' fermarsi per accuratezza target raggiunta.
        early_stop_check = False    # FLAG EVENTO: puo' iniziare il check regolare per l'early stop.
        early_stop = False          # FLAG EVENTO: l'early stop e' scattato, stop dell'addestramento.

        # Con questa lista tengo traccia delle loss da loggare su console come media.
        self.__running_loss = []

        stop_training = False

        # Loop di addestramento per n epoche.
        for current_epoch in range(self.epochs):

            if stop_training:
                break

            # Stop di addestramento. Dimensione batch_size.
            for epoch_step, data in enumerate(self.tr_loader, 0):
                
                cp.cyan(f'Epoch: {current_epoch + 1}, Step: {epoch_step + 1}/{self.tr_steps_per_epoch}')
                
                self.__global_step += 1

                # if self.__global_step == 640:
                #     stop_training = True
                #     break

                # Se l'early stop e' scattato, ci si ferma.
                if early_stop:
                    cp.yellow('Stopping: detected EarlyStop!')
                    stop_training = True
                    break

                # Target performance raggiunta, stop dell'addestramento.
                if target_reached:
                    cp.green('Stopping: accuracy target reached for training and validation!')
                    stop_training = True
                    break

                # L'analisi dell'early stop inizia solo quando sono passati gli step richiesti.
                if (self.__global_step) == es_start_at_step:
                    early_stop_check = True

                # La rete entra in modalità training
                self.net.train()

                # Spacchetto i dati del loader in campioni ed etichette.
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if preview:
                    self.writer.add_images(f'First input batch', inputs)
                    preview = False

                # I campioni viaggiano nella rete.
                outputs = self.net(inputs)

                # Viene calcolata la loss sulle predizioni della rete.
                loss = self.criterion(outputs, labels)

                # Si azzerano i gradienti.
                self.optimizer.zero_grad()

                # Si esegue la backpropagation.
                loss.backward()

                # L'ottimizzatore eseguire il passo di aggiornamento.
                self.optimizer.step()

                self.__running_loss.append(loss.item())

                self.__reset_metrics()

                # Eseguo il log su console.
                if self.__global_step % console_log_step_cadence == 0:
                    self.__console_log(current_epoch, epoch_step)

                # Eseguo il calcolo e il log delle metriche.
                if self.__global_step % compute_metrics_step_cadence == 0:
                    self.__compute_metrics(current_epoch, epoch_step)

                # Controllo dell'early stop:
                # - Se abilitato.
                # - E sono passati gli step richiesti fra un controllo e l'altro.
                if early_stop_check and self.__global_step % early_stop_check_step_cadence == 0:
                    # Se la loss di validazione non migliora da 'patience' volte, e' il momento di alzare il FLAG e richiedere l'early stop.
                    early_stop = self.__check_early_stop(current_epoch, epoch_step)

                # Controllo del raggiungimento del target:
                # - Se sono passati gli step richiesti fra un controllo e l'altro.
                # - Se e' l'ultima epoca di training.
                if self.__global_step == self.last_step or self.__global_step % target_reached_check_step_cadence == 0:
                    # Se entrambe le accuracy hanno raggiunto il target, alzo il FLAG e l'addestramento si fermera'.
                    target_reached = self.__check_target(current_epoch, epoch_step)

        # Salvo l'ultimo stato/modello a termine dell'addestramento.
        torch.save(self.net.state_dict(), self.last_model_outpath_sd)
        torch.save(self.net, self.last_model_outpath)
        cp.yellow('Last model saved.')

        cp.blue('Finished Training.')


    # Resetta alcune metriche per lo step corrente di addestramento.
    def __reset_metrics(self):
        self.__curr_tr_loss = None
        self.__curr_va_loss = None
        self.__curr_tr_acc  = None
        self.__curr_va_acc  = None


    # Verifica il raggiungimento del target.
    def __check_target(self, current_epoch, epoch_step):

        # Calcola accuratezza e confusion matrix per dati di addestramento e validazione.
        self.__curr_tr_acc, tr_conf_matrix = self.compute_accuracy(self.tr_loader, use_current_net = True)
        self.__curr_va_acc, va_conf_matrix = self.compute_accuracy(self.va_loader, use_current_net = True)

        self.__curr_tr_acc *= 100
        self.__curr_va_acc *= 100

        self.writer.add_scalars('accuracy',{'training': self.__curr_tr_acc,'validation': self.__curr_va_acc,}, self.__global_step)
        self.writer.add_figure('confusion matrix/training', tr_conf_matrix, self.__global_step)
        self.writer.add_figure('confusion matrix/validation', va_conf_matrix, self.__global_step)

        cp.yellow(f'CHECKING IF TARGET IS REACHED:')
        cp.yellow(f'Global step: {self.__global_step:5d} - [ep: {current_epoch + 1:3d}, step: {epoch_step + 1:5d}]:')
        cp.yellow(f'Training accuracy  : {self.__curr_tr_acc:.1f}%')
        cp.yellow(f'Validation accuracy: {self.__curr_va_acc:.1f}%')

        tr_improved, va_improved = False, False

        # L'accuracy di training e' migliorata?
        if self.__curr_tr_acc > self.__best_tr_acc:
            self.__best_tr_acc = self.__curr_tr_acc
            tr_improved = True

        # L'accuracy di validazione e' migliorata.
        if self.__curr_va_acc > self.__best_va_acc:
            self.__best_va_acc = self.__curr_va_acc
            va_improved = True

        # Se sono entrambe migliorate, lo considero un buon momento per:
        # - Salvare lo stato di questi pesi.
        # - Considerarli il nuovo MODELLO MIGLIORE.
        if tr_improved and va_improved:
            torch.save(self.net.state_dict(), self.best_model_outpath_sd)
            torch.save(self.net, self.best_model_outpath)
            cp.green('Best model saved.')

        return self.__best_tr_acc > self.__target and self.__best_va_acc > self.__target


    # Verifica la possibilita' di un early stop.
    def __check_early_stop(self, epoch, ep_step):

        self.__curr_va_loss = self.compute_loss(self.va_loader, use_current_net = True) if self.__curr_va_loss is None else self.__curr_va_loss

        cp.cyan("CHECKING EARLY STOP")
        cp.cyan(f'Global step: {self.__global_step:5d} - [ep: {epoch + 1:3d}, step: {ep_step + 1:5d}]')

        # Verifica se lo loss di validazione e' migliorata:
        # - Se non lo e', aggiurno il counter dei NON MIGLIORAMENTI.
        # - Se lo e' ma non a sufficienza, aggiurno il counter dei NON MIGLIORAMENTI.
        # - Se lo e', azzero il counter.
        if self.__curr_va_loss < self.__best_va_loss:

            # Calcolo il tasso di miglioramento.
            improve_ratio = abs((self.__curr_va_loss / self.__best_va_loss) - 1) * 100

            # Verifico che il miglioramento non sia inferiore al tasso.
            if improve_ratio >= self.es_improvement_rate:
                cp.green(f'Validation loss improved: {self.__best_va_loss:.6f} --> {self.__curr_va_loss:.6f} ({improve_ratio:.3f}%)')
                self.__best_va_loss = self.__curr_va_loss
                self.__patience_counter = 0
            else:
                self.__patience_counter += 1
                cp.red(f'Validation loss NOT improved: ({improve_ratio:.3f}%) < ({self.es_improvement_rate}%) ... patience ({self.__patience_counter}/{self.__es_patience})')
        else:
            # Calcolo il tasso di miglioramento.
            improve_ratio = abs((self.__curr_va_loss / self.__best_va_loss) - 1) * 100
            self.__patience_counter += 1
            cp.red(f'Validation loss NOT improved: ({improve_ratio:.3f}%) < ({self.es_improvement_rate}%) ... patience ({self.__patience_counter}/{self.__es_patience})')

        return self.__patience_counter >= self.__es_patience


    # Calcola le metriche in fase di addestramento.
    def __compute_metrics(self, epoch, ep_step):

        # Si calcola la loss sui dati di addestramento e validazione usando la rete corrente.
        self.__curr_tr_loss = self.compute_loss(self.tr_loader, use_current_net = True)
        self.__curr_va_loss = self.compute_loss(self.va_loader, use_current_net = True)

        # Log informazioni sulla tensorboard.
        self.writer.add_scalars('losses',{'training': self.__curr_tr_loss,'validation': self.__curr_va_loss,}, self.__global_step)
        cp.blue("COMPUTING METRICS:")
        cp.blue(f'Global step: {self.__global_step:5d} - [ep: {epoch + 1:3d}, step: {ep_step + 1:5d}]')
        cp.blue(f'Training loss  : {self.__curr_tr_loss:.6f}')
        cp.blue(f'Validation loss: {self.__curr_va_loss:.6f}')


    # Log delle informazioni correnti di addestramento sulla console.
    def __console_log(self, epoch, i):
        current_loss = sum(self.__running_loss) / len(self.__running_loss)
        self.__running_loss.clear()
        print(f'Global step: {self.__global_step:5d} - [ep: {epoch + 1:3d}, step: {i + 1:5d}] - Current loss: {current_loss:.6f}')


    # Calcola la loss sul dataloader fornito e utilzzando la rete specificata.
    def compute_loss(self, loader: torch.utils.data.DataLoader, use_current_net: bool):

        # Se non specifico diversamente, testo sui dati di test.
        if loader is None:
            loader = self.te_loader

        # Se richiesto, testo sul modello corrente e non il migliore.
        if use_current_net:
            net = self.net
        else:
            net = self.__get_net(model_to_load = 'best')

        losses = []

        net.eval()                                              # Rete in modelita' valutazione.
        with torch.no_grad():                                   # Impedito calcolo dei gradienti.
            for inputs, labels in loader:                       # Spacchettamento campioni-etichette.
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = net(inputs)                           # Dati viaggiano nella rete.
                losses.append(self.criterion(outputs, labels))  # Traccia della loss.
        net.train()                                             # Rete in modalita' training.

        # Si restituisce la loss complessiva sui dati.
        return sum(losses) / len(losses)


    # Calcola l'accuratezza sul dataloader fornito e utilzzando la rete specificata.
    def compute_accuracy(self, loader: torch.utils.data.DataLoader, use_current_net: bool):

        # Se non specifico diversamente, testo sui dati di test.
        if loader is None:
            loader = self.te_loader

        # Se richiesto, testo sul modello corrente e non il migliore.
        if use_current_net:
            net = self.net
        else:
            net = self.__get_net(model_to_load = 'best')

        real_y, pred_y = [], []

        net.eval()                                          # Rete in modelita' valutazione.
        with torch.no_grad():                               # Impedito il calcolo dei gradienti.
            for inputs, labels in loader:                   # Spacchettamento campioni-etichette.
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = net(inputs)                       # Dati viaggiano nella rete.
                _, predicted = torch.max(outputs.data, 1)
                real_y = real_y + labels.tolist()
                pred_y = pred_y + predicted.tolist()
        net.train()                                         # Rete in modalita' training.

        # Usa la classe di utilita' per il calcolo delle metriche.
        mt = Metrics(self.classes, real_y, pred_y)

        # Restituisce accuratezza e confusion matrix.
        return mt.accuracy(), mt.get_confusion_matrix_figure()


    # Esegue il test sui campioni.
    def test(self, use_current_net: bool):
        accuracy, te_conf_matrix = self.compute_accuracy(self.te_loader, use_current_net)
        self.writer.add_figure('confusion matrix/test', te_conf_matrix, self.__global_step)
        return accuracy


    # Restituisce l'architettura di rete specificata.
    # - se richiesto 'curr': non carica alcun peso da disco.
    # - se richiesto 'best': carica i pesi migliori della rete salvati su disco.
    # - se richiesto 'last': carica gli ultimi pesi della rete salvati su disco.
    def __get_net(self, model_to_load : str = "curr"):

        # Seleziona la rete.
        if self.cfg.train_parameters.network_type.lower() == 'net_1':
            from nets.net_1 import Net
        elif self.cfg.train_parameters.network_type.lower() == 'net_2':
            from nets.net_2 import Net
        elif self.cfg.train_parameters.network_type.lower() == 'net_3':
            from nets.net_3 import Net
        elif self.cfg.train_parameters.network_type.lower() == 'net_4':
            from nets.net_4 import Net
        else:
            cp.red(f'Unknown net.')
            sys.exit(-1)

        # Inizializza l'architettura.
        net = Net(self.classes)

        # Se richiesto, tenta il caricamento dei pesi migliori.
        if model_to_load.lower() == 'best':
            try:
                net.load_state_dict(torch.load(self.best_model_outpath_sd)).to(self.device)
                cp.green('Best model state_dict successfully reloaded.')
            except:
                cp.red('Missing model state_dict.')
                sys.exit(-1)

        # Se richiesto, tenta il caricamento dei pesi migliori.
        if model_to_load.lower() == 'last':
            try:
                net.load_state_dict(torch.load(self.last_model_outpath_sd)).to(self.device)
                cp.green('Last model state_dict successfully reloaded.')
            except:
                cp.red('Missing model state_dict.')
                sys.exit(-1)

        return net