import torch
import torch.nn as nn


class TextLevelGNN(nn.Module):  
    """
    Text-Level GNN-Modell:
    - Jedes Wort wird als Knoten betrachtet.
    - Zu jedem Knoten gibt es Nachbarn und Kantengewichte.
    - Aus den Nachbarn werden Nachrichten berechnet.
    - Diese Nachrichten werden mit der eigenen Knoteneinbettung kombiniert.
    - Anschließend werden alle Knoten eines Textes zu einer Repräsentation gepoolt
      und für die Klassifikation genutzt.
    """

    def __init__(self, args, embed_pretrained): 
        # Initialisiert den PyTorch Module Parent
        super().__init__()   
        # Hyperparameter definieren
        # ------------------------------------------------------------
        # Anzahl der Knoten im Graphen (Vokabularumfang)
        self.n_node: int = int(args.n_word)
        # Dimension der Embeddings 
        self.d_model: int = int(args.d_model)
        # Anzahl der Zielklassen für die Klassifikation
        self.n_class: int = int(args.n_class)

        # Optionale Modellschalter
        self.use_layer_norm: bool = bool(args.layer_norm)     # LayerNorm nutzen?
        self.use_relu: bool = bool(args.relu)                 # ReLU vor Klassifikation?
        self.mean_reduction: bool = bool(args.mean_reduction) # Nachbarschaften mitteln statt max-poolen?

        # Dropout zur Regularisierung
        self.dropout = nn.Dropout(float(args.dropout))

        # Embedding-Layer
        # ------------------------------------------------------------
        if bool(args.pretrained):
            # Falls vortrainierte Embeddings genutzt werden sollen,
            # prüfen, ob ein gültiger Tensor übergeben wurde: shape [vocab_size, embedding_dim]
            if embed_pretrained is None:
                raise ValueError("args.pretrained=True, aber embeds_pretrained ist None.")
            if embed_pretrained.dim() !=2:
                raise ValueError("embeds_pretrained muss 2D sein")
            if embed_pretrained.size(0) != self.n_node:
               raise ValueError("n_node mismatch")
            if embed_pretrained.size(1) != self.d_model:
                raise ValueError("d_model mismatch")

            # Embedding-Layer aus vortrainierten Embeddings erzeugen
            # freeze=False: die Embeddings bleiben trainierbar.
            # padding_idx=0: Index 0 ist das Padding-Token
            self.embedding = nn.Embedding.from_pretrained(embed_pretrained, freeze = False, padding_idx = 0)

        else:
            # Falls keine vortrainierten Embeddings vorhanden sind:
            # Embeddings zufällig initialisieren
            self.embedding = nn.Embedding(self.n_node, self.d_model, padding_idx=0)    

            # initialisiert einen Tensor in-place mit Normalverteilung(effizient und speichersparend).
            nn.init.normal_(self.embedding.weight, mean = 0.0, std = 0.02)
            # Padding-Vektor (Index 0) explizit auf 0 setzen
            # Keine lernbare Operation
            with torch.no_grad():
                self.embedding.weight[0].fill_(0.0)   

        # Optionale Normalisierung    
        # Wenn aktiviert: LayerNorm auf Embeddings anwenden
        # Sonst: Identity() = keine Veränderung  
        self.ln = nn.LayerNorm(self.d_model) if self.use_layer_norm else nn.Identity()
        
        # Kanten- und Knotengewichte   
        # ------------------------------------------------------------   
        # Anzahl möglicher Edge-IDs
        # +1 padding id(0 wieder für Padding/"keine Kante" reserviert).
        n_edge_ids = (self.n_node - 1) * self.n_node + 1    
        
        # Für jede Kante wird ein skalare Gewicht gelernt
        # Erzeugt ein Embedding-Layer mit: n_edge_ids Einträgen, Ausgabegröße 1
        self.weight_edge = nn.Embedding(n_edge_ids, 1, padding_idx = 0)

        # Mischfaktor
        self.eta_node = nn.Embedding(self.n_node, 1, padding_idx = 0)
                 
        # Init: für skalare Gewichte oft besser klein starten, damit das GNN nicht "explodiert"
        nn.init.zeros_(self.weight_edge.weight)
        nn.init.zeros_(self.eta_node.weight)
        
        # Initialisierung aller Kanten- und Knotengewichte mit 0
        with torch.no_grad():
            self.weight_edge.weight[0].fill_(0.0)
            self.eta_node.weight[0].fill_(0.0)

        # Klassifikationskopf
        # ------------------------------------------------------------
        # Lineare Projektion von d_model -> n_class
        self.fc = nn.Linear(self.d_model, self.n_class, bias = True)
        # Xavier-Initialisierung für stabile Startwerte
        # fc: macht aus der Repräsentation eine Ausgabe
        nn.init.xavier_uniform_(self.fc.weight) 
        nn.init.zeros_(self.fc.bias)


    def forward(self, x, nb_x, w_edge):
        """
        Forward-Pass des Modells.
        - Eingabe: 
            x:[B, L] Wort-IDs des eigentlichen Textes (0 = padding)
            nb_x:[B, L, K] Nachbar-Knoten für jeden Token
            w_edge:[B, L, K] Edge-IDs zwischen Token und seinen Nachbarn
        
        - Ausgabe:
            Scores [B, n_class] Klassifikationsscores für jede Klasse

            B = Anzahl Texte im Batch
            L = Maximale Tokenlänge
            K = Anzahl Nachbarn pro Knoten
            n_class = Ausgabedimension
        """
        
        # Batchgröße und Sequenzlänge extrahieren
        B, L = x.shape
        # Masken erzeugen
        # True für echte Tokens, False für Padding (ID 0)
        mask_nodes = (x != 0).unsqueeze(-1)  # [B, L, 1]
        # True für echte Nachbarn, False für Padding
        mask_nb = (nb_x != 0).unsqueeze(-1)  # [B, L, K, 1]

        # Embeddings laden
        # ------------------------------------------------------------
        # Embeddings der Nachbarn
        emb_nb = self.embedding(nb_x)  # [B, L, K, d_model]
        # Embeddings der Knoten selbst
        emb_self = self.embedding(x)   # [B, L, d_model]

        # Falls aktiviert: LayerNorm auf beide anwenden
        if self.use_layer_norm:
            emb_nb = self.ln(emb_nb)  
            emb_self = self.ln(emb_self)

        # Message für Knoten selbst generieren
        w = self.weight_edge(w_edge) # [B, L, K, 1]
        # Nachbar Message generieren
        msg = w * emb_nb             # [B, L, K, d_model]
         
        # Nachbarn aggregieren
        # ------------------------------------------------------------
        if self.mean_reduction:
            # Padding-Nachbars sollen nicht mitgemittelt werden
            msg = msg.masked_fill(~mask_nb, 0.0)  # Setzt alle Stellen, wo die Maske True ist, auf value
            # Anzahl gültiger Nachbarn je Knoten bestimmen
            denom = mask_nb.sum(dim=2).clamp_min(1) # [B, L, 1]
            msg_nb = msg.sum(dim=2)/denom # [B, L, d_model]

        else:  
            # Max-Pooling
            # Man braucht einen extrem kleinen Wert, damit Padding-Nachbarn beim max niemals gewählt werden.
            # torch.finfo liefert Infos über den Datentyp, .min ist der kleinste darstellbare Wert
            neg_inf = torch.finfo(msg.dtype).min 
            # Fülle Padding-Stellen mit sehr kleinen Wert.
            msg = msg.masked_fill(~mask_nb, neg_inf) 
            # berechnet das Maximum über die Nachbar-Dimension, nimmt nur die Maximalwerte, nicht die Indizes
            msg_nb = msg.max(dim=2).values # [B, L, d_model] 

        # Message Passing 
        # ------------------------------------------------------------
        # eta als Mischfaktor in [0,1], klein ? mehr Nachbarinformation
        eta = torch.sigmoid(self.eta_node(x))  # [B, L, 1] 
        h_node = (1.0 - eta) * msg_nb + eta * emb_self # [B, L, d_model]  
        
        # Pooling über Tokens: ein einziger Vektor für den ganzen Text
        pooled = (h_node * mask_nodes).sum(dim = 1).clamp_min(1)  # [B, d_model]  .clamp_min(1) setzt alle Werte, die kleiner als 1 sind, auf 1
        # Ruft das vorher definierte Dropout-Layer auf
        pooled = self.dropout(pooled)

        # Optional nichtlineare Aktivierung
        if self.use_relu:
            # Setzt alle negativen Werte auf 0
            pooled = torch.nn.functional.relu(pooled) 

        # Klassifikation
        # ------------------------------------------------------------
        # Lineare Projektion auf die Klassen
        scores = self.fc(pooled) # [B, d_model] auf [B, n_class].
        return scores
