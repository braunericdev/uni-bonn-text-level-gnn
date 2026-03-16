import argparse
import pandas as pd
from collections import Counter
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Erstellt label.txt und vocab.txt aus Trainingsdaten.")
    parser.add_argument("dataset_dir", type=str, help="Pfad zum fertigen Ordner (z.B. data/ag_news_converted)")
    parser.add_argument("--min_freq", type=int, default=5, help="Minimale Worthäufigkeit (Standard: 5)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_file = dataset_dir / "train-stemmed.txt"

    if not train_file.exists():
        print(f"❌ Fehler: {train_file} nicht gefunden!")
        return

    print(f"📖 Lese {train_file.name} ein...")
    # Wir lesen die Datei ein. Spalte 0 ist Label, Spalte 1 ist Text
    df = pd.read_csv(train_file, sep='\t', header=None, names=['label', 'text'])

    # --- 1. LABELS GENERIEREN ---
    unique_labels = sorted(df['label'].dropna().unique())
    label_file = dataset_dir / "label.txt"
    
    with open(label_file, "w", encoding="utf-8") as f:
        for label in unique_labels:
            f.write(f"{label}\n")
    print(f"✅ {label_file.name} erstellt! ({len(unique_labels)} verschiedene Klassen gefunden)")

    # --- 2. VOKABULAR GENERIEREN ---
    print(f"⏳ Zähle alle Wörter im Trainingsset (das kann ein paar Sekunden dauern)...")
    word_counter = Counter()
    
    # Durch alle Texte iterieren, am Leerzeichen splitten und zählen
    for text in df['text'].dropna():
        words = str(text).split()
        word_counter.update(words)

    # Nur Wörter behalten, die oft genug vorkommen (min_freq)
    vocab = [word for word, count in word_counter.items() if count >= args.min_freq]
    
    vocab_file = dataset_dir / f"vocab-{args.min_freq}.txt"
    
    with open(vocab_file, "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(f"{word}\n")
            
    print(f"✅ {vocab_file.name} erstellt!")
    print(f"   -> Insgesamt gefundene Wörter: {len(word_counter)}")
    print(f"   -> Davon mindestens {args.min_freq}x vorhanden: {len(vocab)}")

if __name__ == "__main__":
    main()