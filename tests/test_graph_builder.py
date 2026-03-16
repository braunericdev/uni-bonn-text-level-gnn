import numpy as np

from src.graph_builder import compute_valid_edge_ids, build_graph_with_public_edges


def test_graph_construction_with_public_edge():
    """
    Prüft den kompletten Graph-Builder (Sliding Window, Padding, Public Edge)
    anhand eines verständlichen Text-Beispiels.
    """
    # Vokabular-Simulation für ein besseres Verständnis:
    # 1="Ich", 2="mag", 3="Katzen", 4="Hunde", 5="Vögel"

    # Unser Mini-Korpus: Die Kombi "Ich mag" kommt oft vor, die Tiere jeweils nur einmal.
    corpus = [
        [1, 2, 3],  # Satz 1: "Ich mag Katzen"
        [1, 2, 4],  # Satz 2: "Ich mag Hunde"
        [1, 2, 5],  # Satz 3: "Ich mag Vögel"
    ]

    n_word = 5
    n_degree = 1  # p=1: Wir schauen nur 1 Wort nach links und rechts
    k = 2  # Eine Kante muss mind. 2-mal im Korpus auftauchen (Cutoff)
    max_len = 5  # Wir padden alle Sätze auf eine fixe Tensor-Länge von 5

    # Die ID für unsere Public Edge liegt außerhalb des Vokabulars: (5 * 5) + 1 = 26
    public_edge_id = 26

    # 1. Erlaubte Kanten aus dem Korpus extrahieren
    valid_ids = compute_valid_edge_ids(corpus, n_degree, n_word, k)

    # 2. Wir jagen unseren Testsatz "Ich mag Katzen" durch den Builder
    test_text = [1, 2, 3]
    x, nb_x, w_edge = build_graph_with_public_edges(
        test_text, n_degree, max_len, n_word, valid_ids
    )

    # Was erwarten wir mathematisch in der Kanten-Matrix (w_edge)?
    # Zeile 1 ("Ich"): Links nix (0), Rechts "mag" (ID=2) -> valide Kante!
    # Zeile 2 ("mag"): Links "Ich" (ID=6) -> valide!, Rechts "Katzen" -> selten, also Public Edge (26)
    # Zeile 3 ("Katzen"): Links "mag" -> selten (26), Rechts nix (0)
    # Zeile 4 & 5 (Padding): Dürfen keine Kanten haben (0)
    expected_w_edge = np.array(
        [[0, 2], [6, public_edge_id], [public_edge_id, 0], [0, 0], [0, 0]]
    )

    np.testing.assert_array_equal(
        w_edge,
        expected_w_edge,
        err_msg="Graph-Matrix stimmt nicht! Public Edge oder Padding wurde falsch berechnet.",
    )
