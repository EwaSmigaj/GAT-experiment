import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import ToUndirected
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from pandas import DataFrame

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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tworzy mapowania ID na ciągłe indeksy i stosuje je do transakcji.
    Usuwa transakcje z nieprawidłowymi/brakującymi ID.
    """
    print("Mapping IDs")
    pd.set_option('display.max_rows', 20, 'display.max_columns', 100)

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
    
    return users, merchants, transactions


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
    print("Handling String cols for Merchants")

    ## to translate: city and state 

    # encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # cols_to_encode = ["city", "state"]

    # encoded_data = encoder.fit_transform(merchants[cols_to_encode])
    
    # print(f"Encoded data = {encoded_data}")
    
    # one_hot_df = pd.DataFrame(
    #     encoded_data, 
    #     columns=encoder.get_feature_names_out(cols_to_encode),
    #     index=merchants.index 
    # )

    # df_encoded = pd.concat(
    #     [merchants.drop(cols_to_encode, axis=1), one_hot_df], 
    #     axis=1
    # )
    
    # return df_encoded
    merchants =  merchants.drop(columns=["city", "state"])

    return merchants


def handle_string_cols_users(users: pd.DataFrame) -> DataFrame:
    """
    Data columns (total 22 columns):
    #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
    0   user_id                    100 non-null    int64  
    1   name                       100 non-null    object 
    2   current_age                100 non-null    int64  
    3   retirement_age             100 non-null    int64  
    4   birth_year                 100 non-null    int64  
    5   birth_month                100 non-null    int64  
    6   gender                     100 non-null    object 
    7   address                    100 non-null    object 
    8   apartment                  32 non-null     float64
    9   city                       100 non-null    object 
    10  state                      100 non-null    object 
    11  zipcode                    100 non-null    int64  
    12  latitude                   100 non-null    float64
    13  longitude                  100 non-null    float64
    14  per_capita_income_zipcode  100 non-null    float64
    15  yearly_income_person       100 non-null    float64
    16  total_debt                 100 non-null    float64
    17  fico_score                 100 non-null    int64  
    18  num_credit_cards           100 non-null    int64  
    19  debt_to_income_ratio       100 non-null    float64
    20  income_vs_area             100 non-null    float64
    21  years_to_retirement        100 non-null    int64  
    """
    print("Handling String cols for Users")

    ## to translate: city and state 
    users = users.drop(columns=["name", "gender", "address", "city", "state", "apartment"])

    return users


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
    print("Handling String cols for Edges")

    edges["timestamp"] = pd.to_datetime(edges["timestamp"])
    edges["timestamp"] = edges["timestamp"].view(int)//1e9
    
    cols_to_encode = ["card_type"]

    # Dodanie cech binarnych/kategorycznych
    edges['use_chip'] = (edges['use_chip'] == 'Chip Transaction').astype(int)
    edges['has_error'] = edges['has_error'].astype(int)
    edges['card_on_dark_web'] = edges['card_on_dark_web'].astype(int)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    encoded_data = encoder.fit_transform(edges[cols_to_encode])
    
    one_hot_df = pd.DataFrame(
        encoded_data, 
        columns=encoder.get_feature_names_out(cols_to_encode),
        index=edges.index # Important to align indexes for concatenation
    )

    df_encoded = pd.concat(
        [edges.drop(cols_to_encode, axis=1), one_hot_df], 
        axis=1
    )
    
    return df_encoded


def process_node_features(users: pd.DataFrame, merchants: pd.DataFrame) -> HeteroData:
    """
    Przetwarza cechy węzłów i dodaje je do obiektu HeteroData.
    Zwraca wstępnie wypełniony obiekt `data`.
    """
    print("Processing Node features...")

    data = HeteroData()

    # print(f"users")
    # users.info()
    # print(f"MMMMM")
    # merchants.info()
    
    scaler_user = StandardScaler()
    user_features_scaled = scaler_user.fit_transform(users)
    
    data['user'].x = torch.tensor(user_features_scaled, dtype=torch.float)
    data['user'].num_nodes = len(users)

    scaler_merchant = StandardScaler()
    merchant_features_scaled = scaler_merchant.fit_transform(merchants)

    data['merchant'].x = torch.tensor(merchant_features_scaled, dtype=torch.float)
    data['merchant'].num_nodes = len(merchants)
    
    return data


def process_edge_features(data: HeteroData, transactions: pd.DataFrame) -> HeteroData:
    """
    Przetwarza cechy krawędzi, tworzy `edge_index` i dodaje je do `data`.
    """
    print("processing edge features...")
    transactions.info()

    # Sort transactions (for masking in the future)
    transactions = transactions.sort_values('timestamp').reset_index(drop=True)

    edge_index = torch.tensor(
        np.stack([transactions['user_idx'].values, transactions['merchant_idx'].values]), 
        dtype=torch.long
    )

    # Feature cols
    feature_cols = [key for key in list(transactions.columns.values) if key not in ['merchant_idx', 'user_idx', 'card_brand', 'label']]
    edge_attr = torch.tensor(transactions[feature_cols].values, dtype=torch.float)
    edge_label = torch.tensor(transactions['label'].values, dtype=torch.long)

    data['user', 'transacts', 'merchant'].edge_index = edge_index
    data['user', 'transacts', 'merchant'].edge_attr = edge_attr
    data['user', 'transacts', 'merchant'].edge_label = edge_label 
    
    return data

def add_reverse_edges(data):
    edge_index = data[('user', 'transacts', 'merchant')].edge_index
    
    # Create reverse edges
    data[('merchant', 'rev_transacts', 'user')].edge_index = torch.stack([
        edge_index[1],  # merchant nodes
        edge_index[0]   # user nodes
    ])
    
    # Copy edge labels if needed for reverse edges
    if hasattr(data[('user', 'transacts', 'merchant')], 'edge_label'):
        data[('merchant', 'rev_transacts', 'user')].edge_label = \
            data[('user', 'transacts', 'merchant')].edge_label
    
    return data

