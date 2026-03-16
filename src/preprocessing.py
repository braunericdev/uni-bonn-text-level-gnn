import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any


def read_labels(file_path: str | Path) -> Dict[str, int]:
    """
    Liest eine Datei mit Labels (ein Label pro Zeile) und weist jedem Label eine ID zu.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # line.strip() löscht führende und nachfolgende Leerzeichen, sowie Zeilenumbrüche und gibt zurück,
        # ob die Zeile danach leer ist oder nicht.
        # line.strip() = false => Zeile enthält nur Leerzeichen, wird ignoriert
        # splitlines()
        labels = [line.strip() for line in f.read().splitlines() if line.strip()]
        # Erstellt ein Dictionary: {'LabelA': 0, 'LabelB': 1, ...}
        label2idx = {
            # Format des Dictionarys: {Label label: ID i}
            label: i
            # i und label werden Elementen aus der labels zugewiesen
            for i, label
            # enumerate() gibt für jedes Element in labels ein Tupel (Index, Element) zurück
            in enumerate(labels)
        }
    return label2idx


def read_vocab(file_path: str | Path) -> Dict[str, int]:
    """
    Liest eine Datei mit Vokabeln (eine Vokabel pro Zeile) und weist jedem Wort eine ID zu.
    """
    # speichert return von open(...) in f
    with open(file_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f.read().splitlines() if line.strip()]
        # Ertellt Dictionary ab Index 1
        word2idx = {word: i + 1 for i, word in enumerate(words)}
        # das wort <pad> (=Platzhalter) wird in index 0 gespeichert
        #'<pad> ist Konvention, da 'platzhalter' oder 'pad' (z.B. 'iPad') in Texten vorkommen könnten
        word2idx["<pad>"] = 0
        # Falls Unk nicht in Vokabular enhalten gewesen, hier manuell hinzugefügt.
        # Wenn Worte, später nicht im Dict gefunden werden, werden sie als UNK gehandhabt
        if "UNK" not in word2idx:
            word2idx["UNK"] = len(word2idx)
    return word2idx


def read_corpus(
    file_path: str | Path, label2idx: Dict[str, int], word2idx: Dict[str, int]
) -> Tuple[List[List[int]], List[int]]:
    """
    Liest den Textkorpus ein und codiert sowohl die Wörter als auch die Labels.
    Erwartet eine tabulatorgetrennte Datei (Label \t Text).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # splitlines() verhindert leere Strings durch den letzten Zeilenumbruch
        content = [line.split("\t") for line in f.read().splitlines() if line.strip()]

    # data: Jedes Wort im Text (x[1]) wird zu einer ID
    data = [[encode_word(word, word2idx) for word in x[1].split()] for x in content]

    # gt (Ground Truth): Das Label (x[0]) wird zu einer ID
    gt = [label2idx[x[0]] for x in content]

    return data, gt


def encode_word(word: str, word2idx: Dict[str, int]) -> int:
    """Ordnet einem Wort die Id zu"""
    # sucht einfach optimistisch das word im Dict
    try:
        idx = word2idx[word]
    # falls Word nicht gefunden, gib ID für das Wort 'UNK' zurück
    except KeyError:
        idx = word2idx.get("UNK", 0)  # 0, falls auch UNK nicht enthalten
    return idx


def get_embedding(args: Any, word2idx: Dict[str, int]) -> np.ndarray | None:
    """Finde Wörter in den Glove-Embeddings"""
    # suche in Liste 'args' das Argumnet pretrained und wenn dieses false ist ist...
    if not args.pretrained:
        print("\t GLoVe Word-Embeddings gefunden, aber werden nicht verwendet!")
        return None

    # 1. Moderner Dateipfad mit f-String und Pathlib
    # (Setzt voraus, dass args.path_data z.B. "data/" ist und args.d_pretrained z.B. 100)
    file_path = Path(f"{args.path_data}glove.6B.{args.d_pretrained}d.txt")

    # 2. Matrix initialisieren (np.random.uniform)
    vocab_size = len(word2idx)  # Anzahl Zeilen
    embedding_matrix = np.random.uniform(
        -np.sqrt(0.06),
        np.sqrt(0.06),
        (
            vocab_size,  # jedes zeile ein Wort,
            args.d_pretrained,  # jede Spalte eine Dim des Emebeddings
        ),
    )
    emb_counts = 0

    # 3. Datei SICHER öffnen (mit 'with', damit sie danach wieder geschlossen wird)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # entfernt leerzeichen und teilt zeile in d teille (word, word embedding...)
            parts = line.strip().split()
            # speichert den ersten teil der zeile als Wort
            word = parts[0]

            # 4. Wenn die Länge stimmt UND wir das Wort in unserem Datensatz brauchen
            if len(parts) == (args.d_pretrained + 1) and word in word2idx:
                # Vektor extrahieren und an der richtigen Stelle (anhand id des worts) in der Matrix speichern
                vector = np.array([float(x) for x in parts[1:]])
                embedding_matrix[word2idx[word]] = vector
                emb_counts += 1

    # 5. Das <pad> Token auf 0 setzen (Index 0)
    embedding_matrix[0] = np.zeros(args.d_pretrained)

    print(f"\tPretrained GloVe found: {emb_counts}")
    return embedding_matrix
