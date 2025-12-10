import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import ToUndirected
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

# --- Stałe (Configuration) ---
DATA_PATH = "../data/gold/"
RANDOM_STATE = 42
HIDDEN_CHANNELS = 64
EPOCHS = 200
PATIENCE = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
MODEL_FILENAME = 'best_fraud_gnn.pt'


def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def load_df(name: str) -> pd.DataFrame:
        return pd.read_csv(DATA_PATH + name)

    users = load_df("u100_df_nodes_user_clean.csv")
    merchants = load_df("u100_df_nodes_merchant_clean.csv")
    transactions = load_df("u100_df_edges_clean.csv")
    return users, merchants, transactions


def map_and_clean_ids(
    users: pd.DataFrame, 
    merchants: pd.DataFrame, 
    transactions: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Tworzy mapowania ID na ciągłe indeksy i stosuje je do transakcji.
    Usuwa transakcje z nieprawidłowymi/brakującymi ID.
    """
    # 1. Mapowanie ID
    user_id_map = {uid: idx for idx, uid in enumerate(users['user_id'].values)}
    merchant_id_map = {mid: idx for idx, mid in enumerate(merchants['merchant_id'].values)}

    transactions['user_idx'] = transactions['user_id'].map(user_id_map)
    transactions['merchant_idx'] = transactions['merchant_id'].map(merchant_id_map)

    # 2. Czyszczenie ID
    initial_len = len(transactions)
    transactions = transactions.dropna(subset=['user_idx', 'merchant_idx'])
    if len(transactions) < initial_len:
        print(f"Ostrzeżenie: Usunięto {initial_len - len(transactions)} transakcji z brakującymi ID.")

    transactions['user_idx'] = transactions['user_idx'].astype(int)
    transactions['merchant_idx'] = transactions['merchant_idx'].astype(int)
    print(f"Po mapowaniu ID - Ważne transakcje: {len(transactions)}")
    
    return users, merchants, transactions, user_id_map, merchant_id_map


def process_node_features(users: pd.DataFrame, merchants: pd.DataFrame) -> HeteroData:
    """
    Przetwarza cechy węzłów i dodaje je do obiektu HeteroData.
    Zwraca wstępnie wypełniony obiekt `data`.
    """
    data = HeteroData()
    
    # Cechy użytkowników (Users)
    user_cols = [
        'zipcode', 'state', 'latitude', 'longitude',
        'current_age', 'retirement_age', 'address', 'yearly_income_person', 
        'total_debt', 'fico_score', 'num_credit_cards', 
        'debt_to_income_ratio', 'per_capita_income_zipcode', 'income_vs_area'
    ]
    # W oryginalnym kodzie cechy użytkowników nie były skalowane. 
    # Dla ujednolicenia skalowania danych (co jest dobrą praktyką dla GNN):
    user_features = users[user_cols].fillna(0).values
    scaler_user = StandardScaler()
    user_features_scaled = scaler_user.fit_transform(user_features)
    
    data['user'].x = torch.tensor(user_features_scaled, dtype=torch.float)
    data['user'].num_nodes = len(users)

    # Cechy sprzedawców (Merchants)
    merchant_features = merchants[['mcc']].fillna(0).values
    scaler_merchant = StandardScaler()
    merchant_features_scaled = scaler_merchant.fit_transform(merchant_features)
    data['merchant'].x = torch.tensor(merchant_features_scaled, dtype=torch.float)
    data['merchant'].num_nodes = len(merchants)
    
    return data

def handle_string_cols_merchant(merchants: pd.DataFrame) -> DataFrame:
    """
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
    0   merchant_id    35838 non-null  int64  
    1   merchant_name  35838 non-null  int64  
    2   city           35838 non-null  object 
    3   state          35838 non-null  object 
    4   zipcode        35838 non-null  float64
    5   mcc            35838 non-null  int64  
    """

    # to translate: city and state 

    encoder = OneHotEncoder(sparse_output=False)

    merchants["city", "state"] = merchants.fit_transform(merchants["city", "state"])

    return merchants

def handle_string_cols_edge(edges: pd.DataFrame) -> DataFrame:
    """
        Data columns (total 24 columns):
    #   Column                  Non-Null Count    Dtype  
    ---  ------                  --------------    -----  
    0   user_id                 1258233 non-null  int64  
    1   card_id                 1258233 non-null  int64  
    2   merchant_id             1258233 non-null  int64  
    3   card_index              1258233 non-null  int64  
    4   label                   1258233 non-null  int64  
    5   amount                  1258233 non-null  float64
    6   timestamp               1258233 non-null  object 
    7   transaction_hour        1258233 non-null  int64  
    8   day_of_week             1258233 non-null  int64  
    9   use_chip                1258233 non-null  object 
    10  has_error               1258233 non-null  int64  
    11  card_brand              1258233 non-null  object 
    12  card_type               1258233 non-null  object 
    13  card_number             1258233 non-null  int64  
    14  cvv                     1258233 non-null  int64  
    15  has_chip                1258233 non-null  int64  
    16  cards_issued            1258233 non-null  int64  
    17  credit_limit            1258233 non-null  float64
    18  card_on_dark_web        1258233 non-null  int64  
    19  account_age_days        1258233 non-null  int64  
    20  years_since_pin_change  1258233 non-null  int64  
    21  months_until_expiry     1258233 non-null  float64
    22  user_idx                1258233 non-null  int64  
    23  merchant_idx            1258233 non-null  int64 
    """

    # To translate: timestamp, use_chip, card_brand, card_type
    numeric_edges = edges
    numeric_edges["timestamp"] = pd.to_datetime(numeric_edges["timestamp"])
    numeric_edges["timestamp"] = numeric_edges["timestamp"].dt.timestamp().astype(np.int64)

    encoder = OneHotEncoder(sparse_output=False)

    numeric_edges["use_chip", "card_brand", "card_type"] = encoder.fit_transform(numeric_edges["use_chip", "card_brand", "card_type"])

    return numeric_edges


def process_edge_features(data: HeteroData, transactions: pd.DataFrame) -> HeteroData:
    """
    Przetwarza cechy krawędzi, tworzy `edge_index` i dodaje je do `data`.
    """
    # 1. Edge Index
    edge_index = torch.tensor(
        np.stack([transactions['user_idx'].values, transactions['merchant_idx'].values]), 
        dtype=torch.long
    )
    data['user', 'transacts', 'merchant'].edge_index = edge_index

    # 2. Edge Features
    trans_features = transactions[[
        'amount', 'transaction_hour', 'day_of_week', 'credit_limit',
        'account_age_days', 'years_since_pin_change', 'months_until_expiry'
    ]].fillna(0).copy()

    # Dodanie cech binarnych/kategorycznych
    trans_features['use_chip'] = (transactions['use_chip'] == 'Chip Transaction').astype(int)
    trans_features['has_error'] = transactions['has_error'].astype(int)
    trans_features['card_on_dark_web'] = transactions['card_on_dark_web'].astype(int)

    # Skalowanie
    scaler_trans = StandardScaler()
    trans_features_scaled = scaler_trans.fit_transform(trans_features)
    data['user', 'transacts', 'merchant'].edge_attr = torch.tensor(
        trans_features_scaled, dtype=torch.float
    )

    # 3. Edge Labels (Etykiety)
    data['user', 'transacts', 'merchant'].edge_label = torch.tensor(
        transactions['label'].values, dtype=torch.long
    )
    
    return data


def split_edges_and_create_masks(data: HeteroData, transactions: pd.DataFrame) -> HeteroData:
    """
    Dzieli krawędzie na zbiory treningowe, walidacyjne i testowe (Train/Val/Test split)
    i tworzy maski dla PyTorch Geometric.
    """
    n_trans = len(transactions)
    indices = np.arange(n_trans)
    
    # 1. Podział na zbiory
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=transactions['label']
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=RANDOM_STATE, 
        stratify=transactions.iloc[train_idx]['label']
    )

    # 2. Tworzenie masek
    train_mask = torch.zeros(n_trans, dtype=torch.bool)
    val_mask = torch.zeros(n_trans, dtype=torch.bool)
    test_mask = torch.zeros(n_trans, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data['user', 'transacts', 'merchant'].train_mask = train_mask
    data['user', 'transacts', 'merchant'].val_mask = val_mask
    data['user', 'transacts', 'merchant'].test_mask = test_mask

    # 3. Dodanie krawędzi odwróconych (niekierunkowy graf)
    data = ToUndirected()(data)
    
    print(f"\nStruktura grafu po podziale:")
    print(data)
    
    return data


# --- Modele GNN pozostają klasami (są już modułowe) ---

class GNN(torch.nn.Module):
    """Prosty model SAGEConv do generowania osadzeń węzłów."""
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels) # Użycie -1 dla PyG >= 2.4.0
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return x


class HeteroGNN(torch.nn.Module):
    """Model Heterogeniczny GNN dla Link Prediction."""
    def __init__(self, metadata, hidden_channels, num_edge_features, data):
        super().__init__()
        
        # Liniowe projekcje cech węzłów
        self.user_lin = torch.nn.Linear(data['user'].x.shape[1], hidden_channels)
        self.merchant_lin = torch.nn.Linear(data['merchant'].x.shape[1], hidden_channels)
        
        # Model GNN zintegrowany z heterogeniczną warstwą
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata, aggr='sum') 
        
        # Klasyfikator krawędzi
        self.edge_lin = torch.nn.Linear(num_edge_features, hidden_channels)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 3, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, 2)
        )
        
    def forward(self, x_dict, edge_index_dict, edge_attr, edge_index):
        # 1. Projekcja cech
        x_dict = {
            'user': self.user_lin(x_dict['user']).relu(),
            'merchant': self.merchant_lin(x_dict['merchant']).relu()
        }
        
        # 2. Propagacja GNN (generowanie osadzeń)
        x_dict = self.gnn(x_dict, edge_index_dict)
        
        # 3. Ekstrakcja osadzeń dla krawędzi
        src_embeddings = x_dict['user'][edge_index[0]]
        dst_embeddings = x_dict['merchant'][edge_index[1]]
        
        # 4. Przetwarzanie cech krawędzi
        edge_embeddings = self.edge_lin(edge_attr).relu()
        
        # 5. Klasyfikacja
        edge_repr = torch.cat([src_embeddings, dst_embeddings, edge_embeddings], dim=-1)
        return self.classifier(edge_repr)


# --- Funkcje Treningowe i Ewaluacyjne ---

def calculate_class_weights(transactions: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Oblicza wagi klas dla zrównoważenia niezrównoważonego zbioru danych."""
    class_counts = np.bincount(transactions['label'].values)
    # Odwrotna częstość
    class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float).to(device)
    # Normalizacja wag
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    return class_weights


def train_step(model: torch.nn.Module, data: HeteroData, optimizer: torch.optim.Optimizer, class_weights: torch.Tensor) -> float:
    """Wykonuje jeden krok treningowy."""
    model.train()
    optimizer.zero_grad()
    
    edge_type = ('user', 'transacts', 'merchant')
    
    out = model(
        data.x_dict, 
        data.edge_index_dict, 
        data[edge_type].edge_attr, 
        data[edge_type].edge_index
    )
    
    train_mask = data[edge_type].train_mask
    loss = F.cross_entropy(
        out[train_mask], 
        data[edge_type].edge_label[train_mask],
        weight=class_weights
    )
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: HeteroData) -> Dict[str, Dict[str, float]]:
    """Oblicza metryki ewaluacyjne (Acc, Precision, Recall, F1) dla każdego podziału."""
    model.eval()
    edge_type = ('user', 'transacts', 'merchant')
    
    out = model(
        data.x_dict, 
        data.edge_index_dict, 
        data[edge_type].edge_attr, 
        data[edge_type].edge_index
    )
    pred = out.argmax(dim=-1)
    
    results = {}
    for split_mask_name in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[edge_type][split_mask_name]
        y_true = data[edge_type].edge_label[mask]
        y_pred = pred[mask]
        
        acc = (y_pred == y_true).sum() / mask.sum()
        
        # Metryki dla klasy pozytywnej (fraud = 1)
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[split_mask_name] = {
            'acc': float(acc),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results


def run_training_loop(model: torch.nn.Module, data: HeteroData, optimizer: torch.optim.Optimizer, class_weights: torch.Tensor, device: torch.device):
    """Główna pętla treningowa z wczesnym zatrzymaniem i zapisem modelu."""
    print("\nRozpoczynanie treningu GNN...")
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_step(model, data, optimizer, class_weights)
        metrics = evaluate(model, data)
        
        if epoch % 10 == 0:
            print(f'Epoka {epoch:03d}, Strata: {loss:.4f}')
            print(f"  Trening - Acc: {metrics['train_mask']['acc']:.4f}, F1: {metrics['train_mask']['f1']:.4f}")
            print(f"  Walid.  - Acc: {metrics['val_mask']['acc']:.4f}, F1: {metrics['val_mask']['f1']:.4f}")
            print(f"  Test    - Acc: {metrics['test_mask']['acc']:.4f}, F1: {metrics['test_mask']['f1']:.4f}")
        
        # Wczesne zatrzymanie
        if metrics['val_mask']['f1'] > best_val_f1:
            best_val_f1 = metrics['val_mask']['f1']
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_FILENAME)
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\nWczesne zatrzymanie po epoce {epoch}")
            break


def final_evaluation(model: torch.nn.Module, data: HeteroData):
    """Ładuje najlepszy model i wyświetla końcowe wyniki."""
    try:
        model.load_state_dict(torch.load(MODEL_FILENAME))
    except FileNotFoundError:
        print("Nie znaleziono zapisanego modelu. Używanie stanu z ostatniej epoki.")
        
    final_metrics = evaluate(model, data)

    print("\n" + "="*50)
    print("OSTATECZNE WYNIKI (Najlepszy Model)")
    print("="*50)
    for split in ['train_mask', 'val_mask', 'test_mask']:
        split_name = split.replace('_mask', '').capitalize()
        m = final_metrics[split]
        print(f"\nZbiór {split_name}:")
        print(f"  Dokładność (Accuracy): {m['acc']:.4f}")
        print(f"  Precyzja (Precision):  {m['precision']:.4f}")
        print(f"  Czułość (Recall):      {m['recall']:.4f}")
        print(f"  F1 Score:              {m['f1']:.4f}")

# --- Funkcja Główna ---

def main():
    """Główny punkt wejścia do przetwarzania danych i treningu."""
    
    # Ustawienie urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używane urządzenie: {device}")

    ## 1. Przygotowanie danych
    users, merchants, transactions = load_dataframes()
    users, merchants, transactions, _, _ = map_and_clean_ids(users, merchants, transactions)
    
    data = process_node_features(users, merchants)
    data = process_edge_features(data, transactions)
    data = split_edges_and_create_masks(data, transactions)
    data = data.to(device)

    ## 2. Inicjalizacja Modelu i Treningu
    num_edge_features = data['user', 'transacts', 'merchant'].edge_attr.shape[1]
    model = HeteroGNN(data.metadata(), HIDDEN_CHANNELS, num_edge_features, data).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    class_weights = calculate_class_weights(transactions, device)
    
    ## 3. Trening i Ewaluacja
    run_training_loop(model, data, optimizer, class_weights)
    final_evaluation(model, data)


if __name__ == '__main__':
    main()