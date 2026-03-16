# PG Text-Level GNN

## 🚀 Über dieses Projekt

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
* Python 3.10 (aber älter 3.13) ist installiert.
* Poetry ist installiert (`pip install poetry`).

### 2. Projekt klonen & einrichten
Öffne dein Terminal und lade das Projekt herunter:
```bash
git clone [https://gitlab.informatik.uni-bonn.de/projektgruppe2026/pg_text_level_gnn.git](https://gitlab.informatik.uni-bonn.de/projektgruppe2026/pg_text_level_gnn.git)
cd pg_text_level_gnn
```

Erstelle die virtuelle Umgebung und lade alle Pakete in exakt denselben Versionen herunter (dies nutzt automatisch die .lock Datei):
```bash
poetry install
```

### 3. Umgebung aktivieren (Wichtig!)
Damit dein Terminal (und VS Code) die installierten Pakete wie PyTorch findet, musst du in die virtuelle Umgebung wechseln:
```bash
poetry env activate
```
(Alternativ: Wähle in VS Code unten rechts den Python Interpreter aus und klicke auf den Pfad mit .venv im Namen).

## 🌿 Unser Git Workflow
Um Chaos und überschriebenen Code zu vermeiden, nutzen wir den Feature Branch Workflow.

Die goldene Regel: Es wird NIEMALS direkt auf dem main-Branch programmiert! Der main-Branch enthält immer nur funktionierenden, getesteten Code.

Namenskonventionen für Branches
Bitte nutzt für jeden neuen Branch ein passendes Präfix, damit wir wissen, worum es geht:

feature/ für neue Funktionen (z. B. feature/gnn-model)

bugfix/ für Fehlerbehebungen (z. B. bugfix/dataloader-crash)

docs/ für Dokumentation (z. B. docs/readme-setup)

### Der tägliche Ablauf
Neuesten Stand holen:
```bash
git checkout main
git pull origin main
```

Eigenen Branch erstellen:
```bash
git checkout -b feature/dein-feature-name
```

Programmieren & Committen:
```bash
git add .
git commit -m "Beschreibe kurz, was du gemacht hast"
```

Auf GitLab hochladen:
```bash
git push -u origin feature/dein-feature-name
Merge Request: Gehe in GitLab, erstelle einen "Merge Request" für deinen Branch und lass ihn von einem anderen Gruppenmitglied in den main mergen.
```

## 🛠️ Nützliche Befehle (Dev Tools)
Wir haben Tools installiert, damit unser Code einheitlich bleibt. Bitte führe diese regelmäßig aus:

Code formatieren (Black): 
```bash
poetry run black .
```
Code auf Fehler prüfen (Ruff):
```bash
poetry run ruff check .
```
