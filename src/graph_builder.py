import numpy as np
from collections import defaultdict
from typing import List, Tuple


def compute_valid_edge_ids(
    corpus_tokens: List[List[int]], n_degree: int, n_word: int, k: int = 2
) -> np.ndarray:
    """
    Analysiert den Trainingskorpus und zählt die Häufigkeit aller Wort-Kanten.
    Gibt ein Array mit Kanten-IDs zurück, die mindestens k-mal vorkommen.
    Setzt die 'Public Edge' Strategie um (vgl. Huang et al., 2019, Section 2.1).
    """
    edge_freqs = defaultdict(int)

    # 1. Korpus durchlaufen und Nachbarschaften zählen (Sliding Window)
    for text_tokens in corpus_tokens:
        len_text = len(text_tokens)

        for i in range(len_text):
            # Fenster der Größe n_degree (p) in beide Richtungen prüfen
            for d in range(1, n_degree + 1):
                if i - d >= 0:
                    edge_freqs[(text_tokens[i], text_tokens[i - d])] += 1
                if i + d < len_text:
                    edge_freqs[(text_tokens[i], text_tokens[i + d])] += 1

    valid_ids = []

    # 2. Filtern nach Schwellenwert k und Mapping auf 1D-Index
    for (w1, w2), freq in edge_freqs.items():
        if freq >= k:
            # Hash-Trick: 2D-Tupel in eindeutige 1D-ID umwandeln für schnelles Lookup.
            # (w1 - 1) korrigiert den Offset, da Token-ID 0 für Padding reserviert ist.
            edge_id = ((w1 - 1) * n_word) + w2
            valid_ids.append(edge_id)

    return np.array(valid_ids)


def build_graph_with_public_edges(
    text_tokens: List[int],
    n_degree: int,
    max_len_text: int,
    n_word: int,
    valid_edge_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Baut die Adjazenzmatrix (Knoten und Kanten) für ein einzelnes Dokument.
    Vektorisiert für den Dataloader, um Python for-Schleifen-Bottlenecks zu umgehen.
    """
    len_text = len(text_tokens)
    nb_text = []

    # Globale ID für die Public Edge (sicher außerhalb des Vokabular-Raums platziert)
    public_edge_id = (n_word * n_word) + 1

    # --- SCHRITT 1: NACHBARSCHAFTEN SAMMELN (SLIDING WINDOW) ---
    for idx_token in range(len_text):
        nb_front, nb_tail = [], []

        for i in range(n_degree):
            before_idx = idx_token - 1 - i
            nb_front.append(text_tokens[before_idx] if before_idx > -1 else 0)

            after_idx = idx_token + 1 + i
            nb_tail.append(text_tokens[after_idx] if after_idx < len_text else 0)

        nb_text.append(nb_front + nb_tail)

    # --- SCHRITT 2: PADDING FÜR PYTORCH BATCHES ---
    # Feste Tensor-Größen (max_len_text) garantieren.
    x = np.zeros(max_len_text, dtype=int)
    x[: min(len_text, max_len_text)] = np.array(text_tokens)[:max_len_text]

    nb_x = np.zeros((max_len_text, n_degree * 2), dtype=int)
    nb_x[: min(len(nb_text), max_len_text)] = np.array(nb_text)[:max_len_text]

    # --- SCHRITT 3: KANTEN-IDS BERECHNEN ---
    # .reshape(-1, 1) erzeugt einen Spaltenvektor. NumPy Broadcasting addiert
    # den Wert einer Zeile automatisch auf alle Spalten der Matrix nb_x.
    w_edge_head_idx = ((x - 1) * n_word).reshape(-1, 1)
    w_edge = w_edge_head_idx + nb_x

    # --- SCHRITT 4: PUBLIC EDGE ZUWEISUNG (VEKTORISIERT) ---
    # Wir nutzen bitweise Operatoren (| für OR, & für AND, ~ für NOT) auf Matrizen.

    # Maske A: Finde alle Positionen, die reines Padding sind.
    padding_mask = (nb_x == 0) | (x.reshape(-1, 1) == 0)

    # Maske B: C-optimierter Lookup - welche Kanten-IDs sind valide?
    is_valid_edge = np.isin(w_edge, valid_edge_ids)

    # Maske C: Zuweisungs-Logik. Kante bekommt die Public Edge ID, wenn sie
    # selten ist (~is_valid_edge) UND gleichzeitig kein Padding ist (~padding_mask).
    needs_public_edge = (~is_valid_edge) & (~padding_mask)

    # Boolean Indexing: Überschreiben der Matrizen-Felder
    w_edge[needs_public_edge] = public_edge_id

    # Sicherstellen, dass Padding strikt den Kantenwert 0 behält
    w_edge[padding_mask] = 0

    return x, nb_x, w_edge
