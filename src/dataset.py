import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TextGraphDataset(Dataset):
    def __init__(self, config, text_data, labels):
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

    def __len__(self):
        # Sagt PyTorch, wie viele Texte insgesamt da sind
        return self.num_samples

    def __getitem__(self, index):
        """
        Extrahiert einen Datenpunkt und konstruiert den Text-Graphen.
        """
        # Holt den aktuellen Text
        tokens = self.texts[index]
        seq_length = len(tokens)
        
        all_neighbors = []

        # 1. Lokale Nachbarschaften berechnen (Sliding Window)
        # Wir gehen durch jedes einzelne Wort im Text...
        for pos in range(seq_length):
            left_neighbors = []
            right_neighbors = []
            # ...und schauen für jedes Wort 'window_size' Schritte nach links und rechts
            for step in range(1, self.window_size + 1):
                # Linker Kontext
                left_idx = pos - step
                if left_idx >= 0:
                    left_neighbors.append(tokens[left_idx])
                else:
                    left_neighbors.append(0) # Padding
                    
                # Rechter Kontext
                right_idx = pos + step
                if right_idx < seq_length:
                    right_neighbors.append(tokens[right_idx])
                else:
                    right_neighbors.append(0) # Padding

            # Nachbarn der aktuellen Position zusammenführen
            all_neighbors.append(left_neighbors + right_neighbors)

        # 2. Padding auf max_seq_len für einheitliche Batch-Dimensionen
        actual_len = min(seq_length, self.max_seq_len)
        
        node_features = np.zeros(self.max_seq_len, dtype=np.int64)
        node_features[:actual_len] = np.array(tokens)[:actual_len]

        neighbor_features = np.zeros((self.max_seq_len, self.window_size * 2), dtype=np.int64)
        neighbor_features[:actual_len] = np.array(all_neighbors)[:actual_len]

        # 3. Konstruktion der Edge-IDs basierend auf Vokabular-Indizes
        edge_offsets = ((node_features - 1) * self.vocab_size).reshape(-1, 1)
        edges = edge_offsets + neighbor_features
        
        # Ignoriere Kanten für Padding-Tokens
        edges[node_features == 0] = 0

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

    # Dataloader Instanzen erstellen
    loader_train = DataLoader(
        TextGraphDataset(args, data_dict['tr_data'], data_dict['tr_gt']), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=True
    )

    loader_val = DataLoader(
        TextGraphDataset(args, data_dict['val_data'], data_dict['val_gt']), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=False 
    )

    loader_test = DataLoader(
        TextGraphDataset(args, data_dict['te_data'], data_dict['te_gt']), 
        batch_size=args.batch_size,
        num_workers=args.num_worker, 
        collate_fn=prepare_batch, 
        shuffle=False
    )

    return loader_train, loader_val, loader_test, vocab_dict, embeddings_tensor
