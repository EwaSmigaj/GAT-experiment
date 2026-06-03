import pandas as pd
import numpy as np
import torch
import dgl
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn as nn

def build_dgl_graph(transactions_path="../data/brown/credit_card_transactions-ibm_v2.csv", 
                    cards_path="../data/brown/sd254_cards.csv", 
                    users_path="../data/brown/sd254_users.csv", 
                    max_users=100):

    # ── Wczytaj dane ─────────────────────────────────────────────────
    tx    = pd.read_csv(transactions_path)
    cards = pd.read_csv(cards_path)
    users = pd.read_csv(users_path)

    # ── Ogranicz do pierwszych max_users userów ──────────────────────
    if max_users is not None:
        user_ids = sorted(tx['User'].unique())[:max_users]
        tx    = tx[tx['User'].isin(user_ids)].reset_index(drop=True)
        cards = cards[cards['User'].isin(user_ids)].reset_index(drop=True)
        users = users[users.index.isin(user_ids)].reset_index(drop=True)

        print(f"tx = {tx} \n cards = {cards} \n users = {users}")

    # ── 2. Utwórz globalne indeksy ───────────────────────────────────
    # Węzły user: indeks 0..N_users-1
    user_ids   = sorted(tx['User'].unique())
    user2idx   = {u: i for i, u in enumerate(user_ids)}

    # Węzły merchant: indeks 0..N_merchants-1
    merch_ids  = sorted(tx['Merchant Name'].unique())
    merch2idx  = {m: i for i, m in enumerate(merch_ids)}

    # Węzły card: indeks 0..N_cards-1  (klucz: (User, Card))
    card_keys  = list(cards.set_index(['User','CARD INDEX']).index)
    card2idx   = {k: i for i, k in enumerate(card_keys)}

    # Węzły transaction: każdy rząd tx to osobny węzeł
    n_tx = len(tx)



    # ── 3. Krawędzie ────────────────────────────────────────────────
    # user → card  (owns)
    uc_src, uc_dst = [], []
    for _, row in cards.iterrows():
        u = user2idx[row['User']]
        c = card2idx[(row['User'], row['CARD INDEX'])]
        uc_src.append(u); uc_dst.append(c)

    # card → transaction  (made)
    ct_src, ct_dst = [], []
    for t_idx, row in tx.iterrows():
        c = card2idx.get((row['User'], row['Card']), None)
        if c is not None:
            ct_src.append(c); ct_dst.append(t_idx)

    # transaction → merchant  (at)
    tm_src = list(range(n_tx))
    tm_dst = [merch2idx[m] for m in tx['Merchant Name']]

    graph_data = {
        ('user',        'owns',         'card'):        (torch.tensor(uc_src), torch.tensor(uc_dst)),
        ('card',        'owned_by',     'user'):        (torch.tensor(uc_dst), torch.tensor(uc_src)),
        ('card',        'made',         'transaction'): (torch.tensor(ct_src), torch.tensor(ct_dst)),
        ('transaction', 'made_by',      'card'):        (torch.tensor(ct_dst), torch.tensor(ct_src)),
        ('transaction', 'at',           'merchant'):    (torch.tensor(tm_src), torch.tensor(tm_dst)),
        ('merchant',    'received',     'transaction'): (torch.tensor(tm_dst), torch.tensor(tm_src)),
    }

    hg = dgl.heterograph(graph_data)

    # ── 5. Cechy węzłów ─────────────────────────────────────────────
    scaler = StandardScaler()

    # USER features
    user_df = users.set_index('User' if 'User' in users.columns else users.columns[0])
    user_feat_cols = ['Current Age', 'Retirement Age', 'FICO Score', 'Num Credit Cards']
    # TODO: dodaj/usuń kolumny wg potrzeb; zakoduj kategoryczne
    user_feats = scaler.fit_transform(
        users.loc[users.index.isin(user_ids), user_feat_cols]
        .fillna(0).values
    )
    hg.nodes['user'].data['h'] = torch.tensor(user_feats, dtype=torch.float32)

    # CARD features
    card_feat_cols = ['Cards Issued', 'Year PIN last Changed', 'CVV']
    # TODO: zakoduj Card Brand, Card Type itp.
    card_feats = scaler.fit_transform(
        cards[card_feat_cols].fillna(0).values
    )
    hg.nodes['card'].data['h'] = torch.tensor(card_feats, dtype=torch.float32)

    # TRANSACTION features
    tx_feat_cols = ['Year', 'Month', 'Day', 'Amount', 'MCC']
    tx_copy = tx[tx_feat_cols].copy()
    tx_copy['Amount'] = tx_copy['Amount'].replace('[\$,]', '', regex=True).astype(float)
    tx_feats = scaler.fit_transform(tx_copy.fillna(0).values)
    hg.nodes['transaction'].data['h'] = torch.tensor(tx_feats, dtype=torch.float32)

    # MERCHANT features – tylko ID jako embedding (brak dodatkowych cech w danych)
    n_merchants = len(merch_ids)
    hg.nodes['merchant'].data['h'] = torch.zeros(n_merchants, 8)  # placeholder

    # ── 6. Labele na węzłach transaction ────────────────────────────
    labels = (tx['Is Fraud?'].str.strip() == 'Yes').astype(int).values
    hg.nodes['transaction'].data['label'] = torch.tensor(labels, dtype=torch.long)

    # ── 7. Przeprojektuj cechy do wspólnego wymiaru ──────────────────
    proj_dim = 32
    projections = {
        'card':        nn.Linear(3, proj_dim),
        'merchant':    nn.Linear(8, proj_dim),
        'transaction': nn.Linear(5, proj_dim),
        'user':        nn.Linear(4, proj_dim),
    }
    with torch.no_grad():
        for ntype, proj in projections.items():
            hg.nodes[ntype].data['h'] = proj(hg.nodes[ntype].data['h'])


    return hg






