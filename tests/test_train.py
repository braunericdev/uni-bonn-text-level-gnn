from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.train import evaluate, train


class TinyClassifier(nn.Module):
    """
    Sehr kleines Testmodell mit festen, lernbaren Logits.

    Das Modell ignoriert die eigentlichen Eingaben und gibt für jedes Element
    im Batch dieselben Klassenscores zurück. Dadurch lässt sich das Verhalten
    von train() und evaluate() leicht testen.
    """
    def __init__(self):
        super().__init__()

        # Zwei lernbare Scores für eine 2-Klassen-Klassifikation
        # Logits sind die rohen Ausgabewerte eines Modells vor der Softmax-Umwandlung.
        self.logits = nn.Parameter(torch.tensor([2.0, 0.0], dtype=torch.float32))

    def forward(self, x, nb_x, w_edge):
        """
        Gibt für jedes Beispiel im Batch dieselben Logits zurück.

        x, nb_x und w_edge werden hier nur entgegengenommen, damit die
        Signatur zum echten Modell passt.
        """
        batch_size = x.shape[0]

        # Formt [2] zu [1, 2] um und erweitert dann auf [batch_size, 2]
        return self.logits.unsqueeze(0).expand(batch_size, -1)


@pytest.fixture
def args():
    """
    Stellt ein einfaches args-Objekt für die Tests bereit.
    Hier wird nur die CPU als Gerät verwendet.
    """
    return SimpleNamespace(device=torch.device("cpu"))


@pytest.fixture
def batch():
    """
    Erzeugt einen kleinen künstlichen Batch mit:
    - x: Eingabetokens
    - nb_x: Nachbarinformationen
    - w_edge: Kantengewichte
    - y: Zielklassen
    """
    return (
        torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.long),
        torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )


def test_train_updates_parameters_and_returns_batch_metrics(args, batch):
    """
    Testet die Trainingsfunktion:
    - Das Modell soll im Trainingsmodus sein.
    - Der zurückgegebene Loss soll dem erwarteten Cross-Entropy-Loss entsprechen.
    - Die Accuracy soll korrekt berechnet werden.
    - Die Modellparameter sollen nach dem Optimierungsschritt verändert sein.
    """
    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [batch]

    # Merkt sich die Parameter vor dem Training
    before = model.logits.detach().clone()

    # Erwartete Scores vor dem Update für beide Beispiele im Batch
    expected_scores = before.unsqueeze(0).expand(2, -1)

    # Erwarteter Loss auf Basis der ursprünglichen Logits
    expected_loss = torch.nn.functional.cross_entropy(expected_scores, batch[3])

    # Führt genau einen Trainingsdurchlauf aus
    loss_mean, acc = train(args, model, data, optimizer)

    # Modell muss sich im Trainingsmodus befinden
    assert model.training is True

    # Der gemeldete mittlere Loss soll zum erwarteten Loss passen
    assert float(loss_mean) == pytest.approx(float(expected_loss), abs=1e-6)

    # Bei Vorhersage der Klasse 0 für beide Beispiele ist die Accuracy 0.5
    assert float(acc) == pytest.approx(0.5, abs=1e-6)

    # Nach dem Training müssen sich die lernbaren Parameter geändert haben
    assert not torch.equal(model.logits.detach(), before)


def test_evaluate_does_not_update_parameters(args, batch):
    """
    Testet die Auswertungsfunktion:
    - Das Modell soll im Evaluationsmodus sein.
    - Der zurückgegebene Loss soll korrekt sein.
    - Die Accuracy soll korrekt berechnet werden.
    - Die Parameter dürfen nicht verändert werden.
    - Es sollen keine Gradienten berechnet oder gespeichert werden.
    """
    model = TinyClassifier()
    data = [batch]

    # Merkt sich die Parameter vor der Auswertung
    before = model.logits.detach().clone()

    # Erwartete Scores vor der Auswertung
    expected_scores = before.unsqueeze(0).expand(2, -1)

    # Erwarteter Loss
    expected_loss = torch.nn.functional.cross_entropy(expected_scores, batch[3])

    # Führt die Auswertung aus
    loss_mean, acc = evaluate(args, model, data)

    # Modell muss sich im Evaluationsmodus befinden
    assert model.training is False

    # Der gemeldete mittlere Loss soll zum erwarteten Loss passen
    assert float(loss_mean) == pytest.approx(float(expected_loss), abs=1e-6)

    # Wieder wird für beide Beispiele Klasse 0 vorhergesagt -> Accuracy 0.5
    assert float(acc) == pytest.approx(0.5, abs=1e-6)

    # Bei evaluate() dürfen die Parameter nicht verändert werden
    assert torch.equal(model.logits.detach(), before)

    # Es dürfen keine Gradienten hinterlegt sein
    assert model.logits.grad is None