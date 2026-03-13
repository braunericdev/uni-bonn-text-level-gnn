import pytest
import numpy as np
from pathlib import Path

from src.preprocessing import read_labels, read_vocab, encode_word

def test_read_labels(tmp_path):
    # 1. Vorbereitung: Wir erstellen eine Fake-Datei im Speicher
    test_dir = tmp_path / "data"
    test_dir.mkdir()
    label_file = test_dir / "label.txt"
    label_file.write_text("Sport\nMusik\nPolitik", encoding="utf-8")

    # 2. Ausführung: Wir rufen DEINE Funktion auf
    result = read_labels(label_file)
    
    # 3. Prüfung: Stimmt das Ergebnis mit unserer Erwartung überein?
    expected = {"Sport": 0, "Musik": 1, "Politik": 2}
    assert result == expected
    assert len(result) == 3