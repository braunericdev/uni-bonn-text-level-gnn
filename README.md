# PG Text-Level GNN

## 🚀 Über dieses Projekt (Informationen für Recruiter & Tech-Leads)

Dieses Repository implementiert ein Text-Level Graph Neural Network (GNN) zur Dokumentenklassifikation (Natural Language Processing). Das primäre Ziel dieses Projekts ist es, nicht nur ein komplexes ML-Modell zu bauen, sondern von der ersten Zeile an einen starken Fokus auf Software Engineering Best Practices im Machine-Learning-Kontext zu demonstrieren.

Das Projekt wurde von Grund auf mit einer modernen, produktionsnahen Architektur entwickelt, um den typischen "Notebook-Code" zu vermeiden und stattdessen eine saubere, skalierbare Codebase zu schaffen.

**Besondere Highlights der Codebase:**
* **Clean Code & Architektur:** Strikte Trennung von Orchestrierung (`main.py`) und Datenverarbeitung (`src/preprocessing.py`). Sämtliche Funktionen sind modular aufgebaut, ausführlich dokumentiert und mit durchgängigem Type Hinting versehen.
* **Qualitätssicherung (Testing):** Die kritische Daten-Pipeline (Vokabular-Aufbau, GloVe-Embedding-Extraktion) ist durch isolierte Unit-Tests mit `pytest` abgesichert, inklusive Mocking von Dateisystemen.
* **Robustheit:** Sicheres Error-Handling von Edge-Cases (z. B. Fallbacks für Unknown-Tokens `<UNK>` und Padding `<pad>`), modernes Datei-Management (via `pathlib` und Context Managern) und deterministische Reproduzierbarkeit durch zentrales Seed-Management.
* **Modernes Tooling:** Die Abhängigkeiten und isolierten Umgebungen werden sauber über Poetry verwaltet. Für eine konstante Code-Qualität nach PEP-8-Standards sorgen Ruff (Linting) und Black (Formatting).

## 🚀 Setup für Gruppenmitglieder

Wir nutzen **Poetry** für die Verwaltung unserer Pakete und der virtuellen Umgebung. Bitte installiere für dieses Projekt keine Pakete global mit `pip`!

### 1. Voraussetzungen
* **Python >= 3.12** ist installiert.
* **Poetry** ist installiert (`pip install poetry`).

### 2. Projekt klonen & einrichten
Öffne dein Terminal und lade das Projekt herunter:
```bash
git clone [https://gitlab.informatik.uni-bonn.de/projektgruppe2026/pg_text_level_gnn.git](https://gitlab.informatik.uni-bonn.de/projektgruppe2026/pg_text_level_gnn.git)
cd pg_text_level_gnn