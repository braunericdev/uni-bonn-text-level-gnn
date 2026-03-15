from types import SimpleNamespace

import pytest
import torch

from src.model import TextLevelGNN


def make_args(**overrides):
    """
    Erstellt ein Argument-Objekt mit Standardwerten für das Modell.
    Mit **overrides können einzelne Werte überschrieben werden.
    """
    base = {
        "n_word": 8,          # Größe des Vokabulars
        "d_model": 4,         # Dimension der Wort-Embeddings
        "n_class": 3,         # Anzahl der Klassen für die Klassifikation
        "layer_norm": False,  # Layer-Normalisierung deaktiviert
        "relu": False,        # ReLU-Aktivierung deaktiviert
        "mean_reduction": True, # Mittelwert-Aggregation aktiv
        "dropout": 0.0,       # Kein Dropout
        "pretrained": False,  # Keine vortrainierten Embeddings
    }
    base.update(overrides)   # Überschreibt Standardwerte, falls angegeben
    return SimpleNamespace(**base)  # Gibt ein Objekt mit Attributzugriff zurück


def test_random_embedding_initializes_padding_row_to_zero():
    """
    Testet, ob das Embedding korrekt initialisiert wird:
    - Die Embedding-Matrix hat die erwartete Größe.
    - Die erste Zeile (Padding-Token) ist komplett Null.
    """
    model = TextLevelGNN(make_args(), embed_pretrained=None)

    # Prüft die Form der Embedding-Matrix
    assert tuple(model.embedding.weight.shape) == (8, 4)

    # Prüft, ob der Padding-Token (Index 0) nur aus Nullen besteht
    assert torch.equal(model.embedding.weight[0], torch.zeros(4))


def test_forward_returns_finite_class_scores():
    """
    Testet die Forward-Funktion des Modells:
    - Gibt die richtige Ausgabeform zurück
    - Enthält keine NaN- oder Inf-Werte
    """
    model = TextLevelGNN(make_args(), embed_pretrained=None)

    # Eingabetokens (Batchgröße = 2, Satzlänge = 3)
    x = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)

    # Nachbarn jedes Tokens im Graphen
    nb_x = torch.tensor(
        [
            [[2, 3], [1, 0], [0, 0]],
            [[4, 5], [3, 0], [4, 0]],
        ],
        dtype=torch.long,
    )

    # Kanten-Gewichte zwischen Tokens
    w_edge = torch.tensor(
        [
            [[1, 2], [3, 0], [0, 0]],
            [[4, 5], [6, 0], [7, 0]],
        ],
        dtype=torch.long,
    )

    # Forward-Pass durch das Modell
    scores = model(x, nb_x, w_edge)

    # Erwartete Ausgabe: Batchgröße x Anzahl Klassen
    assert tuple(scores.shape) == (2, 3)

    # Prüft, dass alle Werte endlich sind (keine NaN oder Inf)
    assert torch.isfinite(scores).all()


def test_pretrained_shape_mismatch_raises_value_error():
    """
    Testet, ob das Modell einen Fehler wirft,
    wenn vortrainierte Embeddings die falsche Form haben.
    """
    args = make_args(pretrained=True)

    # Falsche Vokabulargröße (7 statt 8)
    wrong_vocab = torch.randn(7, 4)

    # Falsche Embedding-Dimension (5 statt 4)
    wrong_dim = torch.randn(8, 5)

    # Erwartet ValueError wegen falscher Vokabulargröße
    with pytest.raises(ValueError, match="n_node mismatch"):
        TextLevelGNN(args, embed_pretrained=wrong_vocab)

    # Erwartet ValueError wegen falscher Embedding-Dimension
    with pytest.raises(ValueError, match="d_model mismatch"):
        TextLevelGNN(args, embed_pretrained=wrong_dim)