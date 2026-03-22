import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any


def read_labels(file_path: str | Path) -> Dict[str, int]:
    """
    Liest eine Datei mit Labels (ein Label pro Zeile) und weist jedem Label eine ID zu.
    Verhindert Fehler durch versehentliche Duplikate in der Datei.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # Alle Zeilen einlesen und leere Zeilen ignorieren
        labels = [line.strip() for line in f.read().splitlines() if line.strip()]
        
    # Leeres Dictionary starten
    label2idx = {}
    
    # Jedes Label sicher zuweisen (Start bei ID 0)
    for label in labels:
        if label not in label2idx:  # <-- Der Schutzschild für Labels!
            label2idx[label] = len(label2idx)
            
    return label2idx

def read_vocab(file_path: str | Path) -> Dict[str, int]:
    """
    Liest eine Datei mit Vokabeln (eine Vokabel pro Zeile) und weist jedem Wort eine ID zu.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # Alle Zeilen einlesen und leere Zeilen ignorieren
        words = [line.strip() for line in f.read().splitlines() if line.strip()]
        
    # Wir starten direkt mit unserem Padding auf ID 0
    word2idx = {"<pad>": 0}
    
    # Jetzt gehen wir alle eingelesenen Wörter durch
    for word in words:
        if word not in word2idx: # <-- Hier ist der Schutzschild gegen Duplikate!
            # len(word2idx) gibt uns immer exakt die nächste, lückenlose Zahl
            word2idx[word] = len(word2idx)
            
    # Falls UNK nicht im Vokabular enthalten war, hängen wir es ganz ans Ende an
    if "UNK" not in word2idx:
        word2idx["UNK"] = len(word2idx)
        
    return word2idx


def read_corpus(file_path: str | Path, label2idx: Dict[str, int], word2idx: Dict[str, int]) -> Tuple[List[List[int]], List[int]]:
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
        print("\t GLoVe Word-Embeddings werden nicht verwendet!")
        return None

    if getattr(args, "path_embedding", None):
        file_path = Path(args.path_embedding)
    else:
        # Fallback auf das bisherige Standardschema im data-Ordner
        file_path = Path(args.path_data) / f"glove.6B.{args.d_pretrained}d.txt"

    if file_path.is_dir():
        txt_candidates = sorted(file_path.glob("*.txt"))
        if not txt_candidates:
            raise FileNotFoundError(
                f"Keine Embedding-Datei im Verzeichnis gefunden: {file_path}. "
                "Setze --path_embedding auf eine .txt-Datei oder ein Verzeichnis mit genau einer .txt-Datei."
            )
        if len(txt_candidates) == 1:
            file_path = txt_candidates[0]
        else:
            matching_dim = [p for p in txt_candidates if str(args.d_pretrained) in p.stem]
            if len(matching_dim) == 1:
                file_path = matching_dim[0]
            else:
                candidates = ", ".join(str(p.name) for p in txt_candidates[:5])
                raise ValueError(
                    f"Mehrere Embedding-Dateien gefunden in {file_path}: {candidates}. "
                    "Setze --path_embedding auf die gewuenschte .txt-Datei."
                )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Embedding-Datei nicht gefunden: {file_path}. "
            "Setze --path_embedding auf die .txt-Datei deiner GloVe-Vektoren."
        )

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
