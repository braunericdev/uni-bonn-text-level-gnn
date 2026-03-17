import argparse
import logging
import time
import copy
import pickle
from pathlib import Path

import numpy as np
import torch

from src.preprocessing import read_labels, read_vocab, get_embedding, read_corpus, encode_word
from src.dataset import build_dataloaders
from src.model import TextLevelGNN
from src.train import train, evaluate
from src.graph_builder import compute_valid_edge_ids

def main():
    """
    Pipeline:
    - Vorbereitung: 
        - Set Logging, Seed, Path, Device 
        - Set parameter
        - Daten laden
    - Modell bauen
    - Modell trainieren
    - Auf Validierungsdaten prüfen
    - Bestes Modell merken
    - Am Ende auf Testdaten auswerten
    """

    args = parse_args()
    set_seed(args.seed)
    prepare_data(args)
    prepare_paths(args)
    setup_logging(args.path_log)
    args.device = resolve_device(args.device)
    train_model(args)

def prepare_data(args):

    if args.dataset not in ['r8', 'r52', 'ohsumed', "bbc_converted"]:
        raise ValueError('Data {data} not supported, currently supports "r8", "r52", "ohsumed" and "bbc".')

    # read files
    print('\n[info] Dataset:', args.dataset)
    time_start = time.time()

    label2idx = read_labels(args.path_data + args.dataset + '/label.txt')
    word2idx = read_vocab(args.path_data + args.dataset + '/vocab-5.txt')
    args.n_class = len(label2idx)
    args.n_word = len(word2idx)
    print('\tTotal classes:', args.n_class)
    print('\tTotal words:', args.n_word)

    embeds = get_embedding(args, word2idx)

    tr_data, tr_gt = read_corpus(args.path_data + args.dataset + '/train-stemmed.txt', label2idx, word2idx)
    print('\n\tTotal training samples:', len(tr_data))

    # --- START PUBLIC EDGE INTEGRATION ---
    print('\tBerechne valide Kanten (Public Edge Strategie)...')
    valid_edge_ids = compute_valid_edge_ids(tr_data, args.n_degree, args.n_word, k=2)
    # --- ENDE PUBLIC EDGE INTEGRATION ---

    val_data, val_gt = read_corpus(args.path_data + args.dataset + '/valid-stemmed.txt', label2idx, word2idx)
    print('\tTotal validation samples:', len(val_data))

    te_data, te_gt = read_corpus(args.path_data + args.dataset + '/test-stemmed.txt', label2idx, word2idx)
    print('\tTotal testing samples:', len(te_data))

    # save processed data
    mappings = {
        'label2idx': label2idx,
        'word2idx': word2idx,
        'tr_data': tr_data,
        'tr_gt': tr_gt,
        'val_data': val_data,
        'val_gt': val_gt,
        'te_data': te_data,
        'te_gt': te_gt,
        'embeds': embeds,
        'valid_edge_ids': valid_edge_ids,  # <--- HIER ergänzt
        'args': args
    }

    with open(args.path_data + args.dataset + '.pkl', 'wb') as f:
        pickle.dump(mappings, f)

    print('\n[info] Time consumed: {:.2f}s'.format(time.time() - time_start))


def parse_args() -> argparse.Namespace:
    # argparse: ermöglicht, Parameter wie --epochs 100 über die Konsole zu übergeben
    parser = argparse.ArgumentParser(description='Training eines TextLevelGNN Modells zur Textklassifikation')

    # Experiment setting
    # --dataset: Command Line Interface Argument
    parser.add_argument('--dataset', type=str, default='ohsumed', choices=['mr', 'ohsumed', 'r8', 'r52', 'bbc_converted'],
                        help='Name des Datensatzes')
    # --mean_reduction nicht angegeben, dann ist der Wert False
    # --mean_reduction angegeben, dann ist der Wert True
    parser.add_argument('--mean_reduction', action='store_true', help='Ablation: Mean statt Max Aggregation verwenden')
    parser.add_argument('--pretrained', action='store_false', help='Ablation: Keine vortrainierte GloVe Embeddings verwenden')
    parser.add_argument('--layer_norm', action='store_true', help='Ablation: Layer Normalization verwenden')
    parser.add_argument('--relu', action='store_false', help='Ablation: ReLU Aktivierung vor Softmax verwenden')
    parser.add_argument('--n_degree', type=int, default=3, help='Radius der Nachbarschaft im Graph')

    # Hyperparameter
    parser.add_argument('--d_model', type=int, default=300, help='Dimension der Wortrepräsentation')
    parser.add_argument('--d_pretrained', type=int, default=300, help='Dimension der vortrainierten Embeddings')
    parser.add_argument('--max_len_text', type=int, default=100,
                        help='Maximale Länge eines Textes, default 100, 150 für ohsumed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Gerät für Berechnung (cpu oder cuda:0)')

    # Trainingsparameter
    parser.add_argument('--num_worker', type=int, default=5, help='Anzahl Worker für DataLoader')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch Größe')
    parser.add_argument('--epochs', type=int, default=100, help='Maximale Anzahl Trainingsepochen')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Rate (0 = no dropout)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initialisierte Lernrate')
    parser.add_argument('--lr_step', type=int, default=5, help='Nach wie vielen Epochen die Lernrate reduziert wird')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Faktor der Lernratenreduktion')
    parser.add_argument('--es_patience_max', type=int, default=10, help='Geduld (patience) für Early Stopping')
    parser.add_argument('--loss_eps', type=float, default=1e-4, help='Minimale Verbesserung des Loss')
    parser.add_argument('--seed', type=int, default=1111, help='Zufallsseed')

    # Pfade
    parser.add_argument('--path_data', type=str, default='./data/', help='Pfad zum Datensatz')
    parser.add_argument('--path_embedding', type=str, default='./data/glove/', help='Pfad zur GloVe-.txt-Datei oder zu einem Verzeichnis mit Embeddings')
    parser.add_argument('--path_log', type=str, default='./result/logs/', help='Pfad zu training logs')
    parser.add_argument('--path_model', type=str, default='./result/models/', help='Pfad für trainierte Modelle')
    parser.add_argument('--save_model', type=bool, default=True, help='Modelle speichern für weitere Verwendung')

    args = parser.parse_args()
    return args



# Logging konfigurieren
def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# Seeds setzen für Reproduzierbarkeit
def set_seed(seed):
    # Setzt den Seed für NumPy
    np.random.seed(seed)
    # Setzt den Seed für PyTorch auf der CPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Setzt den Seed für alle CUDA-GPUs
        torch.cuda.manual_seed_all(seed)


# Device bestimmen
def resolve_device(device_str):
    # Ob der gewünschte Device-String mit "cuda" beginnt
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA nicht verfügbar, verwende CPU")
        return torch.device("cpu")
    return torch.device(device_str)


# Pfade vorbereiten
def prepare_paths(args):
    # Verzeichnisse aus Argumenten holen
    # Wandelt die übergebenen Pfad-Strings in Path-Objekte um
    data_dir = Path(args.path_data)  
    log_dir = Path(args.path_log)
    model_dir = Path(args.path_model)

    dataset_file = data_dir / f"{args.dataset}.pkl"

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Datensatz nicht gefunden: {dataset_file}. "
            "Bitte preprocessing.py vorher ausführen."
        )
    
    # Log- und Modellordner erstellen
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("_%b_%d_%H_%M")
    """
    %b	abgekürzter Monatsname	Mar
    %d	Tag des Monats	15
    %H	Stunde (24h-Format)	14
    %M	Minute	30
    _Mar_15_14_30    
    """
    args.path_data = str(dataset_file)
    args.path_log = str(log_dir / f"log{timestamp}.txt")
    args.path_model = str(model_dir / f"model{timestamp}.pt")
    args.path_model_params = str(model_dir / f"model_params{timestamp}.pt")


# Modell und Optimizer erstellen
def build_training(args):
    # Datensätze und Embeddings laden mit multiple assignment
    train_loader, valid_loader, test_loader, word2idx, embeds_pretrained = build_dataloaders(args)
    model = TextLevelGNN(args, embeds_pretrained).to(args.device) # verschiebt das Modell auf device
    # Der Optimizer aktualisiert die Modellgewichte während des Trainings.
    optimizer = torch.optim.Adam(
        # nur Parameter trainieren, bei denen requires_grad = True
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,  # learning rate: teuert wie stark Gewichte angepasst werden
        weight_decay=1e-4   # Hilft gegen Overfitting
    )
    # Verändert die Learning Rate während des Trainings
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,  # Der Scheduler steuert die Learning Rate des Optimizers
        step_size=args.lr_step,  # Alle N Epochen wird die Learning Rate veränder
        gamma=args.lr_gamma  # Faktor zur Reduktion
    )

    return model, optimizer, scheduler, train_loader, valid_loader, test_loader


# Training
def train_model(args):

    model, optimizer, scheduler, train_loader, valid_loader, test_loader = build_training(args)

    logging.info("[Run-Konfiguration]")
    logging.info(f" Dataset: {args.dataset}")
    logging.info(f" Logfile: {args.path_log}")
    logging.info(f" Pretrained: {args.pretrained}")
    logging.info(f" Embedding-Pfad: {args.path_embedding if args.path_embedding else '-'}")
    logging.info(f" d_model: {args.d_model} | d_pretrained: {args.d_pretrained}")
    logging.info(f" batch_size: {args.batch_size} | epochs: {args.epochs} | lr: {args.lr}")
    logging.info(f" lr_step: {args.lr_step} | lr_gamma: {args.lr_gamma}")
    logging.info(f" es_patience_max: {args.es_patience_max} | loss_eps: {args.loss_eps}")
    logging.info(f" dropout: {args.dropout} | n_degree: {args.n_degree} | max_len_text: {args.max_len_text}")
    logging.info(f" mean_reduction: {args.mean_reduction} | layer_norm: {args.layer_norm} | relu: {args.relu}")

    logging.info("\n[Training gestartet]")
    # Speichere die bisher beste Validierungsleistung
    loss_best = float("inf")  # Positive Unendlichkeit(“schlechtestmöglichen” großen Wert)
    acc_best = float("-inf")  # Negative Unendlichkeit, damit jede echte Accuracy größer ist
    epoch_best = 0   # Speichert später die beste Epoche
    es_patience = 0  # Zähler für Early Stopping
    # Bestes Modell am Anfang sichern
    # In PyTorch enthält state_dict() die Parameter des Modells (z.B. Bias, Gewicht)
    state_best = copy.deepcopy(model.state_dict()) # echte, unabhängige Kopie

    # Epochenschleife
    for epoch in range(1, args.epochs + 1): # bis stop-1
        logging.info(f"\n[Epoch {epoch}]") # Epochennummer loggen
        start = time.time()  # Startzeit merken
        loss_train, acc_train = train(args, model, train_loader, optimizer) # Ein Trainingsdurchlauf
        scheduler.step()  # Der Learning-Rate-Scheduler wird einen Schritt weitergesetzt
        
        # Trainingswerte loggen
        logging.info(
            f" Train | loss {loss_train:.4f} | acc {acc_train:.4f} | {time.time()-start:.2f}s"
        )

        # Das Modell auf den Validierungsdaten testen
        loss_val, acc_val = evaluate(args, model, valid_loader)

        # Ist das aktuelle Modell besser als das bisher beste?
        # Falls die Accuracy gleich ist, wird zusätzlich die Loss verglichen
        improved = acc_val > acc_best or (
            acc_val == acc_best and (loss_best - loss_val) > args.loss_eps
        )

        if improved:
            # Wenn verbessert: bestes Modell speichern
            state_best = copy.deepcopy(model.state_dict())
            loss_best = loss_val
            acc_best = acc_val
            epoch_best = epoch
            # Setzt Early-Stopping-Zähler zurück, weil es wieder eine Verbesserung gab
            es_patience = 0
        else:
            # Wenn das Modell nicht besser wurde, erhöht sich der Zähler
            es_patience += 1

        # Validierungswerte loggen
        logging.info(
            f" Valid | loss {loss_val:.4f} | acc {acc_val:.4f} "
            f"| patience {es_patience}/{args.es_patience_max}"  
            # Wie viele Epochen ohne Verbesserung vergangen sind
        )

        # Early Stopping prüfen
        if es_patience >= args.es_patience_max:
            # Wenn zu viele Epochen hintereinander keine Verbesserung bringen, 
            # wird das Training abgebrochen
            logging.info("\nEarly Stopping ausgelöst")
            logging.info(
                f"Bestes Modell: Epoch {epoch_best} "
                f"| loss {loss_best:.4f} | acc {acc_best:.4f}"
            )
            break  # for-Schleife beenden


    # Testphase
    logging.info("\n[Testphase]")
    # Lädt die gespeicherten Parameter zurück ins Modell
    model.load_state_dict(state_best)
    # Das beste Modell auf dem Testset bewerten
    loss_test, acc_test = evaluate(args, model, test_loader)
    logging.info(
        f"\n Test | loss {loss_test:.4f} | acc {acc_test:.4f}"
    )

    # Modell speichern (optional)
    if args.save_model:
        # Nur Gewichte speichern
        torch.save(model.state_dict(), args.path_model_params)
        # Komplette Modellstruktur + Gewichte spichern
        torch.save(model, args.path_model)
        logging.info("Modell gespeichert")


if __name__ == "__main__":
    main()
