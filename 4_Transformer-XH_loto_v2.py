#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Polazni kod treba samo da se promeni da radi nad mojim CSV-om, a sintetička od demoa izbaciti. 

Razumeo. Pravilo za sve buduće modele:

polazni kod iz članka se direktno menja da radi nad tvojim loto CSV-om
sintetička demo data, neiskorišćeni delovi, sve što ne pripada polaznom zadatku se izbacuje
predviđa se sledeće loto kolo + back-test, snimanje u TXT
bez paralelnih "novih" klasa ispod polaznog, bez pitanja
"""




"""
Hibridne arhitekture za predikciju koje kombinuju deep learning i klasične time-series modele.

4. Transformer-XH: Hybrid Attention for Long Horizons (PyTorch with PyWavelets)


pip install pywavelets
(pywt se importuje iz paketa PyWavelets)


klase HierarchicalDownsampler, SparseAttention, TransformerXH
učitavanje CSV-a, sekvence, trening sa BCE, predikcija sledećeg kola, back-test, TXT, plt sa loto cross-attention matricom
forecast_len postavljen na N_MAX=39 da bi forecast_head davao 39 logita po broju.


LSTM nad 256 vremenskih koraka, hidden=256
Drugi LSTM nad downsamplovanim (64 koraka), hidden=256
Cross-attention 256x256 nad svim koracima
BATCH=32 → oko 130 batch-eva po epohi
Realno: par sekundi do desetina sekundi po epohi na M1 CPU.
Ako želiš brže, opcije:
HIDDEN_DIM = 128 (umesto 256) — najveća ušteda, model 4x manji
LOOK_BACK = 128 (umesto 256) — sve dvostruko brže
EPOCHS = 50 (umesto 100)
BATCH = 64 (umesto 32) — brže, ali veća memorija

jedna epoha traje ~15 minuta. Za 100 epoha to je ~25 sati na CPU.
Najveći krivac je prvi LSTM nad LOOK_BACK=256 koracima sa hidden=256 — to je ogromna količina sekvencijalnog računanja po batch-u.
Najlakše: smanjiti samo dve stvari:
LOOK_BACK = 128 (umesto 256)
HIDDEN_DIM = 128 (umesto 256)
Procena: ~1-2 min po epohi → ceo trening ~2-3 sata.
Pustiti da radi preko noći (100 epoha x 15 min ≈ 25h).

Ova postavka 10 minuta na MacBook Pro 16G M1.
"""


import torch
import torch.nn as nn
import pywt
import numpy as np

class HierarchicalDownsampler(nn.Module):
    def __init__(self, downsample_rate=4, input_dim=1):
        super().__init__()
        self.downsample_rate = downsample_rate
        # Learnable wavelet filters for adaptive downsampling
        self.wavelet_conv = nn.Conv1d(
            in_channels=input_dim, out_channels=input_dim * 4, kernel_size=downsample_rate, stride=downsample_rate
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            downsampled: [batch, seq_len//rate, features*4]
        """
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        coeffs = self.wavelet_conv(x)  # Approximate wavelet decomposition
        return coeffs.permute(0, 2, 1)

class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=64):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        # Sliding window attention: each token attends to 64 neighbors
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        
    def forward(self, query, key, value):
        # Only compute attention within local windows
        batch_size, seq_len, embed_dim = query.shape
        windows = seq_len // self.window_size
        new_len = windows * self.window_size  # trunkiraj rep ako seq_len nije deljiv sa window_size
        query = query[:, :new_len, :]
        key = key[:, :new_len, :]
        value = value[:, :new_len, :]
        
        # Reshape into windows
        q = query.reshape(batch_size * windows, self.window_size, embed_dim)
        k = key.reshape(batch_size * windows, self.window_size, embed_dim)
        v = value.reshape(batch_size * windows, self.window_size, embed_dim)
        
        attn_out, attn_weights = self.attention(q, k, v)
        return attn_out.reshape(batch_size, new_len, embed_dim), attn_weights

class TransformerXH(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downsampler = HierarchicalDownsampler(config['downsample_rate'], input_dim=config['input_dim'])
        
        # Two-level hierarchy: original + downsampled
        self.encoder_orig = nn.LSTM(
            input_size=config['input_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=1,
            batch_first=True
        )
        
        self.encoder_down = nn.LSTM(
            input_size=config['input_dim'] * 4,  # Wavelet coeffs expand features
            hidden_size=config['hidden_dim'],
            num_layers=1,
            batch_first=True
        )
        
        # Sparse attention on downsampled sequence
        self.sparse_attn = SparseAttention(
            embed_dim=config['hidden_dim'],
            num_heads=8,
            window_size=config['window_size']
        )
        
        # Cross-attention between original and downsampled representations
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config['hidden_dim'],
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.forecast_head = nn.Linear(config['hidden_dim'], config['forecast_len'])
        
    def forward(self, x):
        """
        Args:
            x: Ultra-long sequence [batch, seq_len=10000, features]
        """
        # Level 1: Original resolution LSTM for fine details
        orig_features, _ = self.encoder_orig(x)  # [batch, seq_len, hidden]
        
        # Level 2: Downsampled sequence for long-term context
        downsampled = self.downsampler(x)  # [batch, seq_len//4, features*4]
        down_features, _ = self.encoder_down(downsampled)
        
        # Sparse attention on downsampled (computationally feasible)
        down_features, sparse_weights = self.sparse_attn(
            down_features, down_features, down_features
        )
        
        # Cross-attention: original queries downsampled context
        # Interpolate down_features to match original length
        down_upsampled = torch.nn.functional.interpolate(
            down_features.transpose(1, 2), size=orig_features.size(1), mode='linear'
        ).transpose(1, 2)
        
        cross_out, cross_weights = self.cross_attention(
            orig_features, down_upsampled, down_upsampled
        )
        
        # Forecast from hierarchical representation
        return self.forecast_head(cross_out[:, -1, :]), cross_weights

# =========================
# Loto 7/39 adaptacija (loto7hh_4620_k41.csv) — polazne klase ostaju gore, demo izbačen
# =========================
import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CSV_PATH = "/Users/4c/Desktop/GHQ/KvantniRegresor/loto7hh_4620_k41.csv"
OUT_TXT = Path("/Users/4c/Desktop/GHQ/TimeSeriesModels/4_Transformer-XH_loto_v2_predikcija.txt")

N_MIN, N_MAX = 1, 39
K = 7
LOOK_BACK = 128        # smanjeno sa 256 zbog brzine na CPU
WINDOWS_RF = (20, 50, 100)
BACKTEST_N = 100
VAL_N = 200
EPOCHS = 50            # smanjeno sa 100
BATCH = 64             # povećano sa 32
LR = 1e-3
HIDDEN_DIM = 64       # smanjeno sa 256 (model ~4x manji)
WINDOW_SIZE = 32       # sparse window (downsampled = 32, 32/32 = 1 window)
DOWNSAMPLE_RATE = 4

T0 = time.time()
print()
print("START 4_Transformer-XH_loto_v2", datetime.today())
print()

df = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(df.values, axis=1)
N_total = draws.shape[0]
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")

print(f"CSV: {CSV_PATH}")
print(f"Broj izvlačenja: {N_total}, brojeva po kolu: {K}")
print()


def draws_to_multihot(rows):
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def rolling_features(y_multi):
    cum = np.cumsum(y_multi, axis=0)
    blocks = []
    for w in WINDOWS_RF:
        rolled = np.zeros_like(cum, dtype=np.float32)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        blocks.append(rolled / float(w))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def gap_matrix(rows):
    n = rows.shape[0]
    gap = np.zeros((n, N_MAX), dtype=np.float32)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i, row in enumerate(rows):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in row:
            last_seen[v - 1] = i
    return gap


def make_sequences(features, targets, look_back):
    X, Y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back:i])
        Y.append(targets[i])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def topk_from_scores(scores_1d, k=K):
    s = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d, y_true):
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true, scores):
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick):
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


Y_full = draws_to_multihot(draws)
rolling_raw = rolling_features(Y_full)
gap_raw = gap_matrix(draws)

sum_col = draws.sum(axis=1, keepdims=True).astype(np.float32)
odd_col = (draws % 2 == 1).sum(axis=1, keepdims=True).astype(np.float32)
low_col = (draws <= 19).sum(axis=1, keepdims=True).astype(np.float32)
range_col = (draws.max(axis=1, keepdims=True) - draws.min(axis=1, keepdims=True)).astype(np.float32)
stats_raw = np.concatenate([sum_col, odd_col, low_col, range_col], axis=1)

step_features_raw = np.concatenate([Y_full, rolling_raw, gap_raw, stats_raw], axis=1).astype(np.float32)

START = max(LOOK_BACK, max(WINDOWS_RF))
feature_scaler = StandardScaler()
step_features = step_features_raw.copy()
step_features[START:] = feature_scaler.fit_transform(step_features_raw[START:]).astype(np.float32)
step_features[:START] = feature_scaler.transform(step_features_raw[:START]).astype(np.float32)

X_seq, Y_seq = make_sequences(step_features, Y_full, LOOK_BACK)
X_seq = X_seq[START - LOOK_BACK:]
Y_seq = Y_seq[START - LOOK_BACK:]

n_total = X_seq.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > VAL_N + 200, "Premalo podataka za train/val/back-test."

X_tr, Y_tr = X_seq[:n_train - VAL_N], Y_seq[:n_train - VAL_N]
X_val, Y_val = X_seq[n_train - VAL_N:n_train], Y_seq[n_train - VAL_N:n_train]
X_back, Y_back = X_seq[n_train:], Y_seq[n_train:]
X_next = step_features[-LOOK_BACK:].reshape(1, LOOK_BACK, step_features.shape[1]).astype(np.float32)

INPUT_DIM = X_seq.shape[-1]
print(f"Feature dim: {INPUT_DIM}, LOOK_BACK: {LOOK_BACK}")
print(f"Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Back-test: {X_back.shape[0]}")
print()


# Konfiguracija Transformer-XH-a (polazni config, samo prilagođene dimenzije za loto)
config = {
    'input_dim': INPUT_DIM,
    'hidden_dim': HIDDEN_DIM,
    'downsample_rate': DOWNSAMPLE_RATE,
    'window_size': WINDOW_SIZE,
    'forecast_len': N_MAX  # 39 sigmoid logita po broju 1..39 (umesto 30 dana)
}

model = TransformerXH(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

pos_weight_value = (N_MAX - K) / K
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((N_MAX,), pos_weight_value, dtype=torch.float32))


def make_loader(X, Y, shuffle):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, generator=generator)


train_loader = make_loader(X_tr, Y_tr, shuffle=False)
val_X_t = torch.from_numpy(X_val)
val_Y_t = torch.from_numpy(Y_val)

best_state = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
best_epoch = 0

print("Treniranje Transformer-XH na loto podacima ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    seen = 0
    for xb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += float(loss.detach().cpu()) * xb.size(0)
        seen += xb.size(0)
    train_loss /= max(seen, 1)

    model.eval()
    with torch.no_grad():
        val_logits, _ = model(val_X_t)
        val_loss = float(criterion(val_logits, val_Y_t).detach().cpu())
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
        print(f"epoch {epoch:4d}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  best_epoch={best_epoch}")

final_state = copy.deepcopy(model.state_dict())
print()
print(f"✅ Trening završen. best_epoch={best_epoch}, best_val_loss={best_val_loss:.5f}")
print()


def predict_scores(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(X[s:s + BATCH])
            logits, _ = model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


def evaluate(model, X, Y):
    scores = predict_scores(model, X)
    return scores, avg_hits(scores, Y), safe_auc(Y, scores), safe_lrap(Y, scores)


model.load_state_dict(best_state)
scores_best, h_best, auc_best, lrap_best = evaluate(model, X_back, Y_back)
next_best = predict_scores(model, X_next)[0]
pick_best = topk_from_scores(next_best)

model.load_state_dict(final_state)
scores_final, h_final, auc_final, lrap_final = evaluate(model, X_back, Y_back)
next_final = predict_scores(model, X_next)[0]
pick_final = topk_from_scores(next_final)

ensemble_scores = (scores_best + scores_final) / 2.0
h_ens = avg_hits(ensemble_scores, Y_back)
auc_ens = safe_auc(Y_back, ensemble_scores)
lrap_ens = safe_lrap(Y_back, ensemble_scores)
pick_ens = topk_from_scores((next_best + next_final) / 2.0)

for name, pick in [("TXH_best", pick_best), ("TXH_final", pick_final), ("TXH_ensemble", pick_ens)]:
    assert len(set(pick.tolist())) == K, f"{name} nema 7 jedinstvenih brojeva"
    assert pick.min() >= N_MIN and pick.max() <= N_MAX, f"{name} van opsega"
    assert list(pick) == sorted(pick.tolist()), f"{name} nije sortiran"

print("Predikcija sledeće Loto 7/39 kombinacije:")
print(f"Transformer-XH_best     -> {pick_best.tolist()}  ({describe(pick_best)})")
print(f"Transformer-XH_final    -> {pick_final.tolist()}  ({describe(pick_final)})")
print(f"Transformer-XH_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})")
print()

print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<24} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'Transformer-XH_best':<24} {h_best:>8.3f} {100*h_best/K:>6.1f}% {auc_best:>7.3f} {lrap_best:>7.3f}")
print(f"{'Transformer-XH_final':<24} {h_final:>8.3f} {100*h_final/K:>6.1f}% {auc_final:>7.3f} {lrap_final:>7.3f}")
print(f"{'Transformer-XH_ensemble':<24} {h_ens:>8.3f} {100*h_ens/K:>6.1f}% {auc_ens:>7.3f} {lrap_ens:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


elapsed = time.time() - T0
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N_total}, epochs={EPOCHS}) ---\n")
    f.write(f"Transformer-XH_best     -> {pick_best.tolist()}  ({describe(pick_best)})\n")
    f.write(f"Transformer-XH_final    -> {pick_final.tolist()}  ({describe(pick_final)})\n")
    f.write(f"Transformer-XH_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})\n")
    f.write(
        f"back-test: BEST hits/7={h_best:.3f}, AUC={auc_best:.3f}, LRAP={lrap_best:.3f}; "
        f"FINAL hits/7={h_final:.3f}, AUC={auc_final:.3f}, LRAP={lrap_final:.3f}; "
        f"ENSEMBLE hits/7={h_ens:.3f}, AUC={auc_ens:.3f}, LRAP={lrap_ens:.3f}; "
        f"baseline={7*7/39:.3f}\n"
    )
    f.write(f"elapsed={elapsed:.1f}s\n")

print(f"Snimljeno u: {OUT_TXT}")
print()
print("STOP", datetime.today())
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")


# Visualize cross-attention: which original steps attend to downsampled context (loto sekvenca umesto klimatske)
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    _, attention_matrix = model(torch.from_numpy(X_next))
plt.matshow(attention_matrix[0].detach().numpy()[-100:, :])
plt.title("Cross-Attention: Last 100 Steps vs Downsampled Context (loto)")
plt.savefig('/Users/4c/Desktop/GHQ/TimeSeriesModels/4_transformer_xh_attention.png')
plt.show()



"""
START 4_Transformer-XH_loto_v2 2026-05-25 11:07:26.963255

CSV: /loto7hh_4620_k41.csv
Broj izvlačenja: 4620, brojeva po kolu: 7

Feature dim: 199, LOOK_BACK: 128
Train: 4192, Val: 200, Back-test: 100

Treniranje Transformer-XH na loto podacima ...
epoch    1/50  train_loss=1.13815  val_loss=1.13756  best_epoch=1
epoch   50/50  train_loss=0.92068  val_loss=1.49668  best_epoch=1

✅ Trening završen. best_epoch=1, best_val_loss=1.13756

Predikcija sledeće Loto 7/39 kombinacije:
Transformer-XH_best     -> [10, 14, 15, 22, 23, 29, 37]  (suma=150, neparnih=4/7, niskih(<=19)=3/7, raspon=27)
Transformer-XH_final    -> [15, 17, 23, 29, 31, 34, 37]  (suma=186, neparnih=6/7, niskih(<=19)=2/7, raspon=22)
Transformer-XH_ensemble -> [15, 17, 23, 29, 31, 34, 37]  (suma=186, neparnih=6/7, niskih(<=19)=2/7, raspon=22)

Back-test (poslednjih 100 izvlačenja):
model                      hits/7    hit%     AUC    LRAP
Transformer-XH_best         1.350   19.3%   0.504   0.268
Transformer-XH_final        1.270   18.1%   0.525   0.263
Transformer-XH_ensemble     1.270   18.1%   0.525   0.260
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /4_Transformer-XH_loto_v2_predikcija.txt

plt.savefig('/4_transformer_xh_attention.png')

STOP 2026-05-25 11:18:10.984314
Ukupno vreme: 0:10:44  (644.0 s)
"""
