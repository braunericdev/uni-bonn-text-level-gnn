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
    return {label: i                    #Format des Dictionarys: {Label label: ID i}
            for i, label                #für jedes Label l und seine Index i in der Liste labels
            in enumerate(labels)}   #enumerate() gibt für jedes Element in der Liste 
                                    #labels ein Tupel (Index, Element) zurück 
                                    #und speichert es in i, label