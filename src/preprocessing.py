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
    return word2idx

def read_corpus(
    file_path: str | Path, 
    label2idx: Dict[str, int], 
    word2idx: Dict[str, int]
) -> Tuple[List[List[int]], List[int]]:
    """
    Liest den Textkorpus ein und codiert sowohl die Wörter als auch die Labels.
    Erwartet eine tabulatorgetrennte Datei (Label \t Text).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # splitlines() verhindert leere Strings durch den letzten Zeilenumbruch
        content = [line.split('\t') for line in f.read().splitlines() if line.strip()]

    # data: Jedes Wort im Text (x[1]) wird zu einer ID
    data = [[encode_word(word, word2idx) for word in x[1].split()] for x in content]
    
    # gt (Ground Truth): Das Label (x[0]) wird zu einer ID
    gt = [label2idx[x[0]] for x in content]

    return data, gt
