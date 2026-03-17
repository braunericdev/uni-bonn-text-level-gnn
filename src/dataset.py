import pickle
import numpy as np
import torch
from src.graph_builder import build_graph_with_public_edges
from torch.utils.data import Dataset, DataLoader
from src.graph_builder import build_graph_with_public_edges

class TextGraphDataset(Dataset):
    def __init__(self, config, text_data, labels, valid_edge_ids):
    def __init__(self, config, text_data, labels, valid_edge_ids):
        """
        Initialisierung des Datasets und Speichern der Parameter.
        """
        self.texts = text_data
        self.labels = labels
        self.num_samples = len(labels)

        # Hyperparameter lokal speichern
        self.vocab_size = config.n_word
        self.window_size = config.n_degree
        self.max_seq_len = config.max_len_text
        self.valid_edge_ids = valid_edge_ids  # <--- HIER ergänzt

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Extrahiert einen Datenpunkt und konstruiert den Text-Graphen via graph_builder.
        """
        tokens = self.texts[index]
        
        # --- START PUBLIC EDGE INTEGRATION ---
        x, nb_x, w_edge = build_graph_with_public_edges(
            text_tokens=tokens,
            n_degree=self.window_size,
            max_len_text=self.max_seq_len,
            n_word=self.vocab_size,
            valid_edge_ids=self.valid_edge_ids
        )
        # --- ENDE PUBLIC EDGE INTEGRATION ---

        return x, nb_x, w_edge, self.labels[index]

# [ ... prepare_batch bleibt unangetastet ... ]

def build_dataloaders(args):
    """ 
    Lädt die gepickelten Preprocessing-Daten und initialisiert die Dataloader.
    """
    with open(args.path_data, 'rb') as file:
        data_dict = pickle.load(file)

    vocab_dict = data_dict['word2idx']
    args_prep = data_dict['args']

    # Dimensionen abgleichen
    if args_prep.d_pretrained != args.d_model:
        raise ValueError("Mismatch zwischen Preprocessing Embedding-Dimension und Modell-Dimension.")
        
    args.n_class = args_prep.n_class
    args.n_word = len(vocab_dict) 
    
    # <--- HIER extrahieren wir die berechneten Kanten-IDs
    valid_edge_ids = data_dict['valid_edge_ids']

    # Dataloader Instanzen erstellen (valid_edge_ids übergeben!)
    loader_train = DataLoader(
        TextGraphDataset(args, data_dict['tr_data'], data_dict['tr_gt'], valid_edge_ids), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=True
    )

    loader_val = DataLoader(
        TextGraphDataset(args, data_dict['val_data'], data_dict['val_gt'], valid_edge_ids), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=False 
    )

    loader_test = DataLoader(
        TextGraphDataset(args, data_dict['te_data'], data_dict['te_gt'], valid_edge_ids), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=False
    )

    embeddings_tensor = torch.tensor(data_dict['embeds'], dtype=torch.float32)

    return loader_train, loader_val, loader_test, vocab_dict, embeddings_tensor


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

    embeddings_tensor = None
    if data_dict['embeds'] is not None:
        embeddings_tensor = torch.tensor(data_dict['embeds'], dtype=torch.float32)

        # Dimensionen abgleichen: [vocab_size, embedding_dim]
        if embeddings_tensor.dim() != 2:
            raise ValueError("Preprocessing-Embeddings muessen 2D sein.")
        if embeddings_tensor.size(1) != args.d_model:
            raise ValueError("Mismatch zwischen Preprocessing-Embedding-Dimension und Modell-Dimension.")
    elif bool(args.pretrained):
        raise ValueError("args.pretrained=True, aber im Datensatz sind keine Pretrained-Embeddings gespeichert.")
        
    args.n_class = args_prep.n_class
    args.n_word = len(vocab_dict) 

    # Dataloader Instanzen erstellen (JETZT MIT valid_edges als 4. Parameter!)
    loader_train = DataLoader(
        TextGraphDataset(args, data_dict['tr_data'], data_dict['tr_gt'], data_dict['valid_edge_ids']), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=True

    )

    loader_val = DataLoader(
        TextGraphDataset(args, data_dict['val_data'], data_dict['val_gt'], data_dict['valid_edge_ids']), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=False 
    )

    loader_test = DataLoader(
        TextGraphDataset(args, data_dict['te_data'], data_dict['te_gt'], data_dict['valid_edge_ids']), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=False
    )

    return loader_train, loader_val, loader_test, vocab_dict, embeddings_tensor