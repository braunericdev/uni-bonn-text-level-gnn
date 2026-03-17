import argparse
import pandas as pd
import shutil
import re
from pathlib import Path

def clean_text(text):
    """Macht alles klein, entfernt Zahlen, Satzzeichen und einzelne Buchstaben."""
    text = str(text).lower()
    
    # 1. Behalte AUSSCHLIESSLICH die Buchstaben a bis z (killt Zahlen und Sonderzeichen)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 2. Killt einzelne, alleinstehende Buchstaben (wie das 'd' aus "I'd")
    # \b ist eine Wortgrenze. Sucht nach exakt 1 Buchstaben zwischen Leerzeichen.
    text = re.sub(r'\b[a-z]\b', ' ', text)
    
    # 3. Zieht die übrig gebliebenen doppelten Leerzeichen sauber zusammen
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
    
def main():
    parser = argparse.ArgumentParser(description="Macht aus rohen Parquet-Daten einen perfekten, sauberen GNN-Datensatz.")
    parser.add_argument("dataset_dir", type=str, help="Pfad zum Original-Ordner (z.B. data/ag_news)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = dataset_dir.with_name(f"{dataset_dir.name}_converted")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Zielordner: {output_dir}")

    # --- AG NEWS KATEGORIEN MAPPING ---
    # Parquet speichert oft 0,1,2,3. Wir machen echte Wörter daraus.
    # (Ohne Sonderzeichen wie "/", damit das Modell nicht verwirrt wird)
    label_mapping = {
        0: "world",
        1: "sports",
        2: "business",
        3: "scitech",
        # Falls der Datensatz 1-basiert ist (1,2,3,4):
        4: "scitech" 
    }

    # Hilfsfunktion zum Übersetzen
    def translate_label(label_id):
        try:
            # Falls 1-basiert, ziehen wir 1 ab (1=world, 2=sports...)
            idx = int(label_id)
            if idx > 3 and idx == 4:
                return "scitech"
            elif idx > 0 and idx not in label_mapping: # Falls die IDs 1,2,3,4 sind
                 idx -= 1
            return label_mapping.get(idx, f"class_{idx}")
        except:
            return f"class_{label_id}"

    train_parquet = list(dataset_dir.glob("*train*.parquet"))
    test_parquet = list(dataset_dir.glob("*test*.parquet"))

    if not train_parquet:
        print(f"❌ Keine Train-Parquet Datei gefunden!")
        return

    # --- TRAINING & VALIDIERUNG ---
    print(f"⏳ Verarbeite und wasche {train_parquet[0].name} ...")
    df_train = pd.read_parquet(train_parquet[0])
    
    if 'label' in df_train.columns and 'text' in df_train.columns:
        df_train = df_train[['label', 'text']]
    
    # 1. Labels in Wörter übersetzen!
    df_train.iloc[:, 0] = df_train.iloc[:, 0].apply(translate_label)
    # 2. Texte waschen!
    df_train.iloc[:, 1] = df_train.iloc[:, 1].apply(clean_text)

    valid_df = df_train.sample(frac=0.1, random_state=42)
    train_df = df_train.drop(valid_df.index)

    train_df.to_csv(output_dir / "train-stemmed.txt", sep='\t', index=False, header=False)
    valid_df.to_csv(output_dir / "valid-stemmed.txt", sep='\t', index=False, header=False)

    # --- TEST ---
    if test_parquet:
        print(f"⏳ Verarbeite und wasche {test_parquet[0].name} ...")
        df_test = pd.read_parquet(test_parquet[0])
        
        if 'label' in df_test.columns and 'text' in df_test.columns:
            df_test = df_test[['label', 'text']]
            
        # Hier auch: Labels übersetzen und Texte waschen!
        df_test.iloc[:, 0] = df_test.iloc[:, 0].apply(translate_label)
        df_test.iloc[:, 1] = df_test.iloc[:, 1].apply(clean_text)
        
        df_test.to_csv(output_dir / "test-stemmed.txt", sep='\t', index=False, header=False)

    print("\n✅ Dein Datensatz hat jetzt wörtliche Labels und ist blitzsauber!")

if __name__ == "__main__":
    main()