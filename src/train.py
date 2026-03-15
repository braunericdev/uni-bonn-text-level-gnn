from tqdm import tqdm
import torch 
import torch.nn.functional


def train(args, model, data, optimizer):
    """
    Führt einen Trainingsdurchlauf über den Datensatz aus.
    - args: enthält z.B. device (cpu/gpu)
    - model: das zu trainierende PyTorch Modell
    - data: DataLoader mit Trainingsbatches
    - optimizer: Optimierungsalgorithmus (Adam)
    """
    model.train()  # Modell in den Trainingsmodus setzen (Dropout, BatchNorm aktiv)
    loss_total = 0. # Summe aller Verluste
    n_sample = 0  # Anzahl aller gesehenen Samples
    correct_pred_total = 0  # Anzahl korrekt klassifizierter Samples

    # tqdm erzeugt eine Fortschrittsanzeige für die Batches
    for batch in tqdm(data, desc = "", leave = False):  # leave=False: Balken verschwindet nach Ende der Schleife
        
        """
        batch besteht aus:
        - X: Eingabedaten 
        - NX: Nachbarschaftsinfos
        - EW: Edge Weights
        - Y: Labels/Zielwerte
        """
        X, NX, EW, Y = map(lambda x: x.to(args.device), batch) # Batch auf das gewünschte Gerät verschieben
        # Setzt die gespeicherten Gradienten auf 0 zurück
        # In PyTorch werden Gradienten aufsummiert. Wenn du sie nicht zurücksetzt, würden alte Gradienten mit neuen vermischt.
        optimizer.zero_grad()  
        
        # Forward Pass
        scores_batch = model(X, NX, EW)  
        # Berechnet den Cross-Entropy-Loss für Klassifikation.
        loss_batch = torch.nn.functional.cross_entropy(scores_batch, Y)
        # Backpropagation: Berechnet die Gradienten aller trainierbaren Parameter im Modell
        loss_batch.backward() 
        # Parameterupdate: Der Optimizer aktualisiert die Modellgewichte anhand der berechneten Gradienten
        optimizer.step() 

        # Berechne Loss
        # Der Loss pro Batch wird mit der Batchgröße multipliziert, später den Durchschnitt-Loss pro Sample berechnen kann
        loss_total += loss_batch * scores_batch.shape[0]
        # Anzahl Samples erhöhen
        n_sample += scores_batch.shape[0]
        # Anzahl korrekter Vorhersagen berechnen
        # Sucht entlang der letzten Dimension das Maximum, für jedes Sample die Klasse mit dem höchsten Score finden
        correct_pred_total += (scores_batch.max(dim = -1)[1]==Y).sum() # Vergleicht Vorhersage mit echtem Label, True oder False
        loss_mean = loss_total/n_sample
        acc = correct_pred_total/n_sample

    return loss_mean, acc
    

def evaluate(args, model, data): 
    """
    Evaluierung des Modells auf Validierungs- oder Testdaten.
    """
    # Modell in den Evaluationsmodus setzen
    model.eval()
    loss_total = 0.
    n_sample = 0
    correct_pred_total = 0
    
    # Deaktiviert Gradientberechnung: schneller und speichersparender
    with torch.no_grad():
        for batch in tqdm(data, desc= "evaluating", leave = False):
            # Batch auf Device verschieben
            X, NX, EW, Y = map(lambda x: x.to(args.device), batch)
            # Forward Pass
            scores_batch = model(X, NX, EW)
            # Loss berechnen
            loss_batch = torch.nn.functional.cross_entropy(scores_batch, Y)
            loss_total += loss_batch * scores_batch.shape[0]
            n_sample += scores_batch.shape[0]
            correct_pred_total += (scores_batch.max(dim = -1)[1] == Y).sum()
            loss_mean = loss_total/n_sample
            acc = correct_pred_total/n_sample

        return loss_mean, acc