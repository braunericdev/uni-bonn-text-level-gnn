import pickle
import numpy as np
import torch
from src.graph_builder import build_graph_with_public_edges
from torch.utils.data import Dataset, DataLoader

class TextGraphDataset(Dataset):
    def __init__(self, config, text_data, labels, valid_edge_ids):
        """
        Initialisierung des Datasets und Speichern der Parameter.
        """
        self.texts = text_data
        self.labels = labels
        self.num_samples = len(labels)

        # Hyperparameter lokal speichern
        #wie viele verschiedene Wörter das Modell kennt
        self.vocab_size = config.n_word
        # n_degree ist das "Sliding Window"
        self.window_size = config.n_degree
        self.max_seq_len = config.max_len_text
        self.valid_edge_ids = valid_edge_ids # <--- NEU

    def __len__(self):
        # Sagt PyTorch, wie viele Texte insgesamt da sind
        return self.num_samples

    def __getitem__(self, index):
        """
        Extrahiert einen Datenpunkt und konstruiert den Text-Graphen.
        """
        tokens = self.texts[index]
        
        # Aufruf unserer ausgelagerten, schnellen Profi-Funktion!
        node_features, neighbor_features, edges = build_graph_with_public_edges(
            text_tokens=tokens,
            n_degree=self.window_size,
            max_len_text=self.max_seq_len,
            n_word=self.vocab_size,
            valid_edge_ids=self.valid_edge_ids
        )

        return node_features, neighbor_features, edges, self.labels[index]


def prepare_batch(batch_items):
    """
    Custom collate_fn für den Dataloader.
    Konvertiert die Datenpunkte eines Batches in PyTorch Tensoren.
    """
    nodes, neighbors, edges, targets = zip(*batch_items)

    nodes_tensor = torch.tensor(np.array(nodes), dtype=torch.long)
    neighbors_tensor = torch.tensor(np.array(neighbors), dtype=torch.long)
    edges_tensor = torch.tensor(np.array(edges), dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    
    return nodes_tensor, neighbors_tensor, edges_tensor, targets_tensor


def build_dataloaders(args):
    """ 
    Lädt die gepickelten Preprocessing-Daten und initialisiert die Dataloader.
    """
    with open(args.path_data, 'rb') as file:
        data_dict = pickle.load(file)

    vocab_dict = data_dict['word2idx']
    args_prep = data_dict['args']
    
    # NEU: Wir holen uns die validen Kanten aus dem gepickelten Dictionary
    valid_edges = data_dict['valid_edge_ids'] 

    # Dimensionen abgleichen
    if args_prep.d_pretrained != args.d_model:
        raise ValueError("Mismatch zwischen Preprocessing Embedding-Dimension und Modell-Dimension.")
        
    args.n_class = args_prep.n_class
    args.n_word = len(vocab_dict) 

    # Dataloader Instanzen erstellen (JETZT MIT valid_edges als 4. Parameter!)
    loader_train = DataLoader(
        TextGraphDataset(args, data_dict['tr_data'], data_dict['tr_gt'], valid_edges), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=True
    )

    loader_val = DataLoader(
        TextGraphDataset(args, data_dict['val_data'], data_dict['val_gt'], valid_edges), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=False 
    )

    loader_test = DataLoader(
        TextGraphDataset(args, data_dict['te_data'], data_dict['te_gt'], valid_edges), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=False
    )

    embeddings_tensor = torch.tensor(data_dict['embeds'], dtype=torch.float32)

    return loader_train, loader_val, loader_test, vocab_dict, embeddings_tensor