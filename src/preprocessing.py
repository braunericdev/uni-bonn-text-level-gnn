from pathlib import Path
from typing import Dict


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
            label: i  # Format des Dictionarys: {Label label: ID i}
            for i, label in enumerate(  # für jedes Label l und seine Index i in der Liste labels
                labels
            )  # enumerate() gibt für jedes Element in der Liste
        }  # labels ein Tupel (Index, Element) zurück
        # und speichert es in i, label
    return label2idx


def read_vocab(file_path: str | Path) -> Dict[str, int]:
    """
    Liest eine Datei mit Vokabeln (eine Vokabel pro Zeile) und weist jedem Wort eine ID zu.
    """
    with open(
        file_path, "r", encoding="utf-8"
    ) as f:  # speichert return von open(...) in f
        words = [line.strip() for line in f.read().splitlines() if line.strip()]
        word2idx = {  # Ertellt Dictionary ab Index 1
            word: i + 1 for i, word in enumerate(words)
        }
        word2idx["<pad>"] = (
            0  # das wort <pad> (=Platzhalter) wird in index 0 gespeichert
        )
    return word2idx  #'<pad> ist Konvention, da 'platzhalter' oder 'pad' (z.B. 'iPad') in Texten vorkommen könnten
