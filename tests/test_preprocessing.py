import pytest
import numpy as np
from pathlib import Path

from src.preprocessing import read_labels, read_vocab, encode_word, read_corpus, get_embedding

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

def test_read_vocab(tmp_path):
    # 1. Fake-Datei erstellen
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("Haus\nHund", encoding="utf-8")

    # 2. Ausführen
    result = read_vocab(vocab_file)
    
    # 3. Prüfen: <pad> muss 0 sein, Wörter ab 1, UNK am Ende
    assert result["<pad>"] == 0
    assert result["Haus"] == 1
    assert result["Hund"] == 2
    assert result["UNK"] == 3
    assert len(result) == 4

def test_encode_word():
    word2idx = {"<pad>": 0, "Haus": 1, "UNK": 2}
    
    # Wort ist bekannt -> ID 1
    assert encode_word("Haus", word2idx) == 1
    # Wort ist unbekannt -> Fallback auf UNK (ID 2)
    assert encode_word("Auto", word2idx) == 2
    # Wort und UNK fehlen -> Fallback auf 0
    assert encode_word("Auto", {"<pad>": 0}) == 0

def test_read_corpus(tmp_path):
    corpus_file = tmp_path / "corpus.txt"
    # Format: Label \t Text
    corpus_file.write_text("Sport\tHaus Hund\nMusik\tHaus", encoding="utf-8")
    
    label2idx = {"Sport": 0, "Musik": 1}
    word2idx = {"<pad>": 0, "Haus": 1, "Hund": 2, "UNK": 3}
    
    data, gt = read_corpus(corpus_file, label2idx, word2idx)
    
    # Ground Truth Labels checken
    assert gt == [0, 1]
    # Checken, ob die Wörter richtig in IDs übersetzt wurden
    assert data == [[1, 2], [1]]

def test_get_embedding(tmp_path):
    # Wir bauen ein "Fake"-args Objekt, um das Terminal zu simulieren
    class Args:
        pretrained = True
        path_data = str(tmp_path) + "/"
        d_pretrained = 2  # Nur 2 Dimensionen zum Testen!
        
    args = Args()
    word2idx = {"<pad>": 0, "Haus": 1, "Hund": 2, "UNK": 3}
    
    # Wir erstellen eine winzige Fake-GloVe Datei
    glove_file = tmp_path / "glove.6B.2d.txt"
    glove_file.write_text("Haus 0.5 -0.5\nKatze 1.0 1.0", encoding="utf-8")
    
    matrix = get_embedding(args, word2idx)
    
    # Check 1: Hat die Matrix die richtige Größe? (4 Wörter, 2 Dimensionen)
    assert matrix.shape == (4, 2)
    # Check 2: Ist <pad> wirklich genau 0?
    assert np.array_equal(matrix[0], [0.0, 0.0])
    # Check 3: Wurden die Werte für "Haus" korrekt eingelesen?
    assert np.array_equal(matrix[1], [0.5, -0.5])