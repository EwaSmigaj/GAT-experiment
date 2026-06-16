import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
from sklearn.preprocessing import StandardScaler


def _money(series: pd.Series) -> np.ndarray:
    """'$1,234.56' -> 1234.56"""
    return (series.astype(str)
                  .str.replace(r'[\$,]', '', regex=True)
                  .astype(float)
                  .values)


def build_dgl_graph(transactions_path="../data/brown/credit_card_transactions-ibm_v2.csv",
                     cards_path="../data/brown/sd254_cards.csv",
                     users_path="../data/brown/sd254_users.csv",
                     max_users=50):

    # ── Wczytaj dane ─────────────────────────────────────────────────
    tx    = pd.read_csv(transactions_path)
    cards = pd.read_csv(cards_path)
    users = pd.read_csv(users_path)
    # W tym datasecie indeks wiersza users.csv == wartość 'User' w tx/cards.
    # NIE filtrujemy/resetujemy users, żeby to mapowanie zostało zachowane.

    # ── Ogranicz do pierwszych max_users userów ──────────────────────
    if max_users is not None:
        keep = sorted(tx['User'].unique())[:max_users]
        tx    = tx[tx['User'].isin(keep)].reset_index(drop=True)
        cards = cards[cards['User'].isin(keep)].reset_index(drop=True)

    user_ids  = sorted(tx['User'].unique())
    user2idx  = {u: i for i, u in enumerate(user_ids)}

    merch_ids   = sorted(tx['Merchant Name'].unique())
    merch2idx   = {m: i for i, m in enumerate(merch_ids)}
    n_merchants = len(merch_ids)

    card_keys = list(cards.set_index(['User', 'CARD INDEX']).index)
    card2idx  = {k: i for i, k in enumerate(card_keys)}

    n_tx = len(tx)

    # ── Pre-processing kolumn transakcji ──────────────────────────────
    tx = tx.copy()
    tx['AmountNum'] = _money(tx['Amount'])
    tx['Hour']      = tx['Time'].str.split(':').str[0].astype(float)
    tx['HasError']  = tx['Errors?'].notna().astype(float)

    user_state = users['State']  # index = User ID
    tx['LocMismatch'] = (
        tx['Merchant State'].notna()
        & (tx['Merchant State'] != tx['User'].map(user_state))
    ).astype(float)

    chip_cats = ['Swipe Transaction', 'Online Transaction', 'Chip Transaction']
    use_chip = pd.get_dummies(pd.Categorical(tx['Use Chip'], categories=chip_cats)).astype(float)

    # ── Krawędzie ──────────────────────────────────────────────────────
    uc_src, uc_dst = [], []
    for _, row in cards.iterrows():
        uc_src.append(user2idx[row['User']])
        uc_dst.append(card2idx[(row['User'], row['CARD INDEX'])])

    ct_src, ct_dst = [], []
    for t_idx, row in tx.iterrows():
        c = card2idx.get((row['User'], row['Card']))
        if c is not None:
            ct_src.append(c)
            ct_dst.append(t_idx)

    tm_src = list(range(n_tx))
    tm_dst = [merch2idx[m] for m in tx['Merchant Name']]

    hg = dgl.heterograph({
        ('user',        'owns',     'card'):        (torch.tensor(uc_src), torch.tensor(uc_dst)),
        ('card',        'owned_by', 'user'):        (torch.tensor(uc_dst), torch.tensor(uc_src)),
        ('card',        'made',     'transaction'): (torch.tensor(ct_src), torch.tensor(ct_dst)),
        ('transaction', 'made_by',  'card'):        (torch.tensor(ct_dst), torch.tensor(ct_src)),
        ('transaction', 'at',       'merchant'):    (torch.tensor(tm_src), torch.tensor(tm_dst)),
        ('merchant',    'received', 'transaction'): (torch.tensor(tm_dst), torch.tensor(tm_src)),
    })

    scaler = StandardScaler()

    # ── USER ───────────────────────────────────────────────────────────
    u = users.iloc[user_ids]
    user_feats = np.column_stack([
        u['Current Age'].values,
        u['Retirement Age'].values,
        u['FICO Score'].values,
        u['Num Credit Cards'].values,
        _money(u['Per Capita Income - Zipcode']),
        _money(u['Yearly Income - Person']),
        _money(u['Total Debt']),
        (u['Gender'] == 'Male').astype(float).values,
    ])
    hg.nodes['user'].data['h'] = torch.tensor(
        scaler.fit_transform(np.nan_to_num(user_feats)), dtype=torch.float32)

    # ── CARD ──────────────────────────────────────
    brand_cats = ['Visa', 'Mastercard', 'Amex', 'Discover']
    type_cats  = ['Credit', 'Debit', 'Debit (Prepaid)']

    # Parsowanie dat na same lata (wygodne dla modelowania)
    acct_year = pd.to_datetime(cards['Acct Open Date'], format='%m/%Y').dt.year.values
    expiry_year = pd.to_datetime(cards['Expires'], format='%m/%Y').dt.year.values

    # Konwersja kolumn tekstowych/binarnych na 0.0 / 1.0
    has_chip = (cards['Has Chip'].str.strip().str.upper() == 'YES').astype(float).values
    on_dark_web = (cards['Card on Dark Web'].str.strip().str.lower() == 'yes').astype(float).values

    # Cechy ciągłe do przepuszczenia przez StandardScaler
    card_num = np.column_stack([
        cards['Cards Issued'].values,
        _money(cards['Credit Limit']),
        cards['Year PIN last Changed'].values,
        acct_year,
        expiry_year
    ])
    card_num_scaled = scaler.fit_transform(np.nan_to_num(card_num))

    # One-hot encoding marek i typów kart
    brand_oh = pd.get_dummies(pd.Categorical(cards['Card Brand'], categories=brand_cats)).astype(float).values
    type_oh  = pd.get_dummies(pd.Categorical(cards['Card Type'], categories=type_cats)).astype(float).values

    # Połączenie wszystkiego: wyskalowane numeryczne + kategoryczne + binarne
    card_feats = np.hstack([
        card_num_scaled, 
        brand_oh, 
        type_oh, 
        has_chip[:, None], 
        on_dark_web[:, None]
    ])
    
    hg.nodes['card'].data['h'] = torch.tensor(card_feats, dtype=torch.float32)
    
    # ── MERCHANT ──────────
    tx['MerchIdx'] = tx['Merchant Name'].map(merch2idx)
    stats = tx.groupby('MerchIdx').agg(
        n_tx=('AmountNum', 'count'),
        mean_log_amt=('AmountNum', lambda s: np.log1p(s).mean()),
        std_log_amt=('AmountNum', lambda s: np.log1p(s).std()),
        is_online=('Merchant State', lambda s: s.isna().mean()),
        mcc=('MCC', lambda s: s.mode().iloc[0]),
    ).reindex(range(n_merchants)).fillna(0)

    merch_feats = scaler.fit_transform(np.column_stack([
        np.log1p(stats['n_tx'].values),
        stats['mean_log_amt'].values,
        stats['std_log_amt'].values,
        stats['is_online'].values,
        stats['mcc'].values,
    ]))
    hg.nodes['merchant'].data['h'] = torch.tensor(merch_feats, dtype=torch.float32)

    # ── TRANSACTION ────────────────────────────────────────────────────
    tx_num = np.column_stack([
        tx['Year'].values,
        tx['Month'].values,
        tx['Day'].values,
        tx['Hour'].values,
        np.log1p(tx['AmountNum'].values),
        tx['MCC'].values,
    ])
    tx_feats = np.hstack([
        scaler.fit_transform(np.nan_to_num(tx_num)),
        use_chip.values,
        tx[['HasError', 'LocMismatch']].values,
    ])
    hg.nodes['transaction'].data['h'] = torch.tensor(tx_feats, dtype=torch.float32)

    # ── Labele ──────────────────────────────────────────────────────────
    labels = (tx['Is Fraud?'].str.strip() == 'Yes').astype(int).values
    hg.nodes['transaction'].data['label'] = torch.tensor(labels, dtype=torch.long)


    for ntype in hg.ntypes:
        hg.nodes[ntype].data['h_raw'] = hg.nodes[ntype].data['h']
        del hg.nodes[ntype].data['h']


    return hg