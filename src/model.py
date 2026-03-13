import torch
from torch import nn
import torch.nn.functional

class TextLevelGNN(nn.Module):  # Die Klasse erbt von torch.nn.Module
    # Spreichere args im Modell
    # Token-Embeddings anlegen aus vortrainierten Vektoren oder neu zufällig initialisiert
    # Klassifikationsschicht anlegen. fc: macht aus der Repräsentation eine Ausgabe
    def __init__(self, args, embed_pretrained):  # Hyperparameter, Optionale vortrainierte Wortvektoren (z.B. GloVe)
        super().__init__()   # Initialisiert den PyTorch Module Parent
        self.n_node: int = int(args.n_word)  # Anzahl der Knoten im Graphen, Vokabularumfang
        self.d_model: int = int(args.d_model)  # Embedding-Dimension. Länge des Wortvektors
        self.n_class: int = int(args.n_class) 

        self.use_layer_norm: bool = bool(args.layer_norm)  
        self.use_relu: bool= bool(args.relu) 
        self.mean_reduction: bool = bool(args.mean_reduction) 

        self.dropout = nn.Dropout(float(args.dropout))  

        # Fall: vortrainierte Embeddings: shape [vocab_size , embedding_dim]
        if bool(args.pretrained):
            if embed_pretrained is None:
                raise ValueError("args.pretrained=True, aber embeds_pretrained ist None.")
            if embed_pretrained.dim() !=2:
                raise ValueError("embeds_pretrained muss 2D sein")
            if embed_pretrained.size(0) != self.n_node:
               raise ValueError("n_node mismatch")
            if embed_pretrained.size(1) != self.d_model:
                raise ValueError("d_model mismatch")

            # freeze=False bedeutet: die Embeddings bleiben trainierbar.
            self.embedding = nn.Embedding.from_pretrained(embed_pretrained, freeze = False, padding_inx = 0) # index 0 ist padding

        else:
            self.embedding = nn.Embedding(self.n_node, self.d_model, padding_idx=0)    
            # Embeddings: normal init ist oft stabiler als xavier auf embeddings 
            # initialisiert einen Tensor in-place mit Normalverteilung.
            nn.init.normal_(self.embedding.weight, mean = 0.0, std = 0.02) # _ am Ende bedeutet Objekt wird direkt verändert.
            with torch.no_grad():
                self.embedding.weight[0].fill_(0.0)    # padding vector auf 0
        # Optional normalization        
        self.ln = nn.LayerNorm(self.d_model) if self.use_layer_norm else nn.Identity()
        # Graph parameters      
        # (n_node-1)*n_node kann riesig sein. Behalte ich hier als drop-in replacement. 
        # Man will hier Parameter manuell setzen, ohne dass Autograd das als Teil des Rechengraphen behandelt.
        n_edge_ids = (self.n_node - 1) * self.n_node + 1    # +1 padding id Hier wird berechnet, wie viele mögliche Edge-IDs es gibt.
        # Erzeugt ein Embedding-Layer mit: n_edge_ids Einträgen, Ausgabegröße 1
        self.weight_edge = nn.Embedding(n_edge_ids, 1, padding_idx = 0)

        # Knoten-Self-Weight (eta)
        self.eta_node = nn.Embedding(self.n_node, 1, padding_idx = 0)
                 
        # Init: für skalare Gewichte oft besser klein starten, damit das GNN nicht "explodiert"
        nn.init.zeros_(self.weight_edge.weight)
        nn.init.zeros_(self.eta_node.weight)
        # Ensure padding rows are exactly zero (important!)
        # Innerhalb dieses Blocks werden keine Gradienten aufgezeichnet.
        with torch.no_grad():
            self.weight_edge.weight[0].fill_(0.0)
            self.eta_node.weight[0].fill_(0.0)

        # Classifier head
        # Das ist die finale lineare Schicht.
        self.fc = nn.Linear(self.d_model, self.n_class, bias = True)
        # Initialisiert die Gewichtsmatrix mit Xavier-Uniform-Verteilung.
        nn.init.xavier_uniform_(self.fc.weight)
        # Setzt den Biasvektor auf 0.
        nn.init.zeros_(self.fc.bias)


    def forward(self, x, nb_x, w_edge):
        """
        Eingabe: 
        x:(B, L) node ids/tokens (0 = padding)
        nb_x:(B, L, K) Nachbar idx
        w_edge:(B, L, K)  w_edge:(B, L, K) edge ids aligned with nb_x (0 = padding)
        Ausgabe:
        Scores (B, n_class)

        B = Batch Size
        L = Sequenzlänge / Anzahl Tokens
        K = Anzahl Nachbarn pro Knoten
        n_class = Ausgabedimension
        """
        
        # x.shape liefert die Dimensionen von x, diese zwei Werte werden in B und L entpackt
        B, L = x.shape
        # masks
        mask_nodes = (x != 0).unsqueeze(-1)  # (B, L, 1)
        mask_nb = (nb_x != 0).unsqueeze(-1)  # (B, L, K, 1)

        # Embedding-Layer
        emb_nb = self.embedding(nb_x)  # nb_x enthält Token-/Knoten-IDs, Ergebnis: Vektoren statt IDs
        emb_self = self.embedding(x) # für Knoten selbst

        # Optional Layer Norm
        if self.use_layer_norm:
            # LayerNorm kann direkt auf dem last-dim D laufen, kein view/reshape nötig
            emb_nb = self.ln(emb_nb)  # self.ln(...) ruft entweder LayerNorm oder Identity auf, je nachdem wie self.ln definiert wurde.
            emb_self = self.ln(emb_self)

        # Message generieren
        w = self.weight_edge(w_edge) # (B, L, K, 1)
        # Nachbar Message generieren
        msg = w * emb_nb # (B, L, K, D)
         
        # Nachbarn aggregieren
        if self.mean_reduction:
            # Padding-Nachbars sollen nicht mitgemittelt werden
            msg = msg.masked_fill(~mask_nb, 0.0)  # etzt alle Stellen, wo die Maske True ist, auf value
            denom = mask_nb.sum(dim=2).clamp_min(1) # (B, L, 1)
            msg_nb = msg.sum(dim=2)/denom # (B, L, D)

        else:  # Max-Pooling
            # Man braucht einen extrem kleinen Wert, damit Padding-Nachbarn beim max niemals gewählt werden.
            neg_inf = torch.finfo(msg.dtype).min #torch.finfo(msg.dtype) liefert Infos über den Datentyp, .min ist der kleinste darstellbare Wert
            msg = msg.masked_fill(~mask_nb, neg_inf) # Padding-Stellen werden mit einem sehr kleinen Wert gefüllt.
            msg_nb = msg.max(dim=2).values # (B, L, D) berechnet das Maximum über die Nachbar-Dimension, nimmt nur die Maximalwerte, nicht die Indizes

        # Message Passing 
        # eta als Gate in [0,1] stabiler als unbeschränkt, klein → mehr Nachbarinformation
        eta = torch.sigmoid(self.eta_node(x))  # (B, L, 1) holt pro Knoten einen skalaren Wert, transformiert ihn in den Bereich (0,1)
        h_node = (1.0 - eta) * msg_nb + eta * emb_self # (B, L, D)  Lineare Mischung aus zwei Tensoren
        
        
        # Pooling über Tokens: Aus allen Knotenvektoren eines Textes wird ein einziger Vektor für den ganzen Text gemacht
        # h_node * mask_nodes maskiert Padding-Tokens weg
        pooled = (h_node * mask_nodes).sum(dim = 1).clamp_min(1)  # (B, D)  .clamp_min(1) setzt alle Werte, die kleiner als 1 sind, auf 1
        # Ruft das vorher definierte Dropout-Layer auf. Regularisierung auf dem Textvektor.
        pooled = self.dropout(pooled)
        if self.use_relu:
            pooled = torch.nn.functional.relu(pooled) # Setzt alle negativen Werte auf 0.

        scores = self.fc(pooled) #Projiziert den finalen Textvektor (B, D) auf (B, n_class).
        return scores






