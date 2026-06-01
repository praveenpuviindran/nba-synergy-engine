import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from v3_neural_synergy.synergy_utils import (
    ContextAwareSynergyNet,
    attach_player_quality_column,
    build_player_feature_table,
    build_player_vector_dict,
    featurize_core_candidate,
    parse_group_ids,
)

# -------------------------
# CONFIG
# -------------------------
RANDOM_SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 240
PATIENCE = 28
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CORE_PRIOR_STRENGTH = 4.0  # Empirical-Bayes shrinkage toward global mean.

LINEUPS_PATH = 'v3_neural_synergy/data/lineups_2014_2025.csv'
ARCHETYPES_PATH = 'data/gmm_archetypes.csv'

MODEL_PATH = 'v3_neural_synergy/synergy_model.pth'
ARTIFACT_PATH = 'v3_neural_synergy/synergy_artifacts.pkl'

# -------------------------
# REPRODUCIBILITY
# -------------------------
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print('--- TRAINING CONTEXT-AWARE SYNERGY ENGINE ---')


# -------------------------
# LOAD + PREP PLAYER FEATURES
# -------------------------
lineups = pd.read_csv(LINEUPS_PATH)
archetypes = pd.read_csv(ARCHETYPES_PATH)

# Build player quality priors from historical lineup impact.
# This keeps recommendations from over-indexing on style fit while ignoring skill level.
quality_sum = defaultdict(float)
quality_minutes = defaultdict(float)

for _, row in lineups.iterrows():
    minutes = float(row.get('MIN', 0.0))
    if minutes <= 0:
        continue
    ids = parse_group_ids(row['GROUP_ID'])
    if len(ids) != 5:
        continue
    season = row['SEASON_LABEL']
    net48 = (float(row['PLUS_MINUS']) / minutes) * 48.0
    for pid in ids:
        key = (int(pid), season)
        quality_sum[key] += net48 * minutes
        quality_minutes[key] += minutes

quality_raw = {
    key: quality_sum[key] / quality_minutes[key]
    for key in quality_sum
    if quality_minutes[key] > 0
}

season_to_vals = defaultdict(list)
for (pid, season), value in quality_raw.items():
    season_to_vals[season].append(value)

season_quality_stats = {}
for season, vals in season_to_vals.items():
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if std < 1e-6:
        std = 1.0
    season_quality_stats[season] = (mean, std)

player_quality_z = {}
for key, value in quality_raw.items():
    season = key[1]
    mean, std = season_quality_stats[season]
    player_quality_z[key] = float((value - mean) / std)

archetypes = attach_player_quality_column(archetypes, player_quality_z, col_name='QUALITY_PRIOR_Z')

player_table, tracking_scaler, archetype_ids, player_feature_cols = build_player_feature_table(
    archetypes,
    extra_numeric_cols=['QUALITY_PRIOR_Z'],
)
player_vectors = build_player_vector_dict(player_table, player_feature_cols)

# -------------------------
# BUILD RAW EXAMPLES
# Each lineup yields 5 supervised examples:
#   Input = (core-4 context, candidate-1)
#   Label = lineup net points per 48 minutes
# -------------------------
X_raw = []
y_net48 = []
core_keys = []
sample_weights = []
season_labels = []
candidate_quality = []

skipped_missing_players = 0
skipped_bad_rows = 0

for _, row in lineups.iterrows():
    minutes = float(row.get('MIN', 0.0))
    if minutes <= 0:
        skipped_bad_rows += 1
        continue

    ids = parse_group_ids(row['GROUP_ID'])
    if len(ids) != 5:
        skipped_bad_rows += 1
        continue

    season = row['SEASON_LABEL']

    vectors = []
    for pid in ids:
        vec = player_vectors.get((int(pid), season))
        if vec is None:
            vectors = []
            break
        vectors.append(vec)

    if len(vectors) != 5:
        skipped_missing_players += 1
        continue

    net48 = (float(row['PLUS_MINUS']) / minutes) * 48.0
    weight = np.sqrt(minutes)

    for cand_idx in range(5):
        core_indices = [i for i in range(5) if i != cand_idx]
        core_vecs = np.stack([vectors[i] for i in core_indices], axis=0)
        cand_vec = vectors[cand_idx]
        feature_vec = featurize_core_candidate(core_vecs, cand_vec)

        core_ids = tuple(sorted(int(ids[i]) for i in core_indices))
        key = (season, core_ids)

        X_raw.append(feature_vec)
        y_net48.append(net48)
        core_keys.append(key)
        sample_weights.append(weight)
        season_labels.append(season)
        candidate_quality.append(float(player_quality_z.get((int(ids[cand_idx]), season), 0.0)))

X_raw = np.array(X_raw, dtype=np.float32)
y_net48 = np.array(y_net48, dtype=np.float32)
sample_weights = np.array(sample_weights, dtype=np.float32)
season_labels = np.array(season_labels)
candidate_quality = np.array(candidate_quality, dtype=np.float32)

print(f'Built {len(X_raw)} examples from {len(lineups)} lineups.')
print(f'Skipped rows (bad format): {skipped_bad_rows}')
print(f'Skipped rows (missing player-season vectors): {skipped_missing_players}')

# -------------------------
# SPLIT: train / val / test
# -------------------------
indices = np.arange(len(X_raw))
train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.15,
    random_state=RANDOM_SEED,
    stratify=season_labels,
)

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.18,  # 0.18 * 0.85 ~= 15% overall
    random_state=RANDOM_SEED,
    stratify=season_labels[train_val_idx],
)

# -------------------------
# TRAIN-ONLY CORE BASELINES
# y_delta = lineup_net48 - shrunk(core_mean_net48)
# This forces model to learn candidate-specific marginal fit.
# -------------------------
core_sum = defaultdict(float)
core_count = defaultdict(int)

for idx in train_idx:
    key = core_keys[idx]
    core_sum[key] += float(y_net48[idx])
    core_count[key] += 1

global_train_mean = float(y_net48[train_idx].mean())


def shrunk_core_baseline(key):
    """Return the Empirical-Bayes shrunk mean net-48 for a (season, core_ids) key.

    Unseen cores fall back to the global training mean.
    """
    cnt = core_count.get(key, 0)
    if cnt == 0:
        return global_train_mean
    mean_val = core_sum[key] / cnt
    return (cnt * mean_val + CORE_PRIOR_STRENGTH * global_train_mean) / (cnt + CORE_PRIOR_STRENGTH)


def make_delta_targets(idxs):
    """Compute marginal-fit targets by subtracting each core's shrunk baseline from raw net-48."""
    baselines = np.array([shrunk_core_baseline(core_keys[i]) for i in idxs], dtype=np.float32)
    return y_net48[idxs] - baselines


y_train = make_delta_targets(train_idx)
y_val = make_delta_targets(val_idx)
y_test = make_delta_targets(test_idx)

q_train = candidate_quality[train_idx]
q_val = candidate_quality[val_idx]
q_test = candidate_quality[test_idx]

# -------------------------
# SCALE ENGINEERED INPUTS
# -------------------------
input_scaler = StandardScaler()
X_train = input_scaler.fit_transform(X_raw[train_idx]).astype(np.float32)
X_val = input_scaler.transform(X_raw[val_idx]).astype(np.float32)
X_test = input_scaler.transform(X_raw[test_idx]).astype(np.float32)

w_train = sample_weights[train_idx]
w_val = sample_weights[val_idx]
w_test = sample_weights[test_idx]

# Normalize weights to avoid unstable gradients.
w_train = w_train / (w_train.mean() + 1e-8)
w_val = w_val / (w_val.mean() + 1e-8)
w_test = w_test / (w_test.mean() + 1e-8)

# -------------------------
# TORCH DATALOADERS
# -------------------------
train_ds = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train).view(-1, 1),
    torch.from_numpy(w_train).view(-1, 1),
)
val_ds = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val).view(-1, 1),
    torch.from_numpy(w_val).view(-1, 1),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# TRAIN NEURAL MODEL
# -------------------------
model = ContextAwareSynergyNet(input_dim=X_train.shape[1])
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.SmoothL1Loss(reduction='none')

best_state = None
best_val = float('inf')
patience_left = PATIENCE

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_loss_num = 0.0
    train_loss_den = 0.0

    for xb, yb, wb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss_vec = criterion(pred, yb)
        loss = (loss_vec * wb).sum() / wb.sum()
        loss.backward()
        optimizer.step()

        train_loss_num += float((loss_vec * wb).sum().item())
        train_loss_den += float(wb.sum().item())

    model.eval()
    val_loss_num = 0.0
    val_loss_den = 0.0
    with torch.no_grad():
        for xb, yb, wb in val_loader:
            pred = model(xb)
            loss_vec = criterion(pred, yb)
            val_loss_num += float((loss_vec * wb).sum().item())
            val_loss_den += float(wb.sum().item())

    train_loss = train_loss_num / max(train_loss_den, 1e-8)
    val_loss = val_loss_num / max(val_loss_den, 1e-8)

    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}')

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        patience_left = PATIENCE
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f'Early stopping at epoch {epoch}. Best val loss={best_val:.4f}')
            break

if best_state is not None:
    model.load_state_dict(best_state)

# -------------------------
# FIT KNN + BLEND
# -------------------------
knn = KNeighborsRegressor(n_neighbors=35, weights='distance', metric='minkowski', p=2)
knn.fit(X_train, y_train)

with torch.no_grad():
    pred_val_nn = model(torch.from_numpy(X_val)).squeeze(1).numpy()
    pred_test_nn = model(torch.from_numpy(X_test)).squeeze(1).numpy()

pred_val_knn = knn.predict(X_val)
pred_test_knn = knn.predict(X_test)

best_alpha = 1.0
best_blend_rmse = float('inf')
for alpha in np.linspace(0.0, 1.0, 21):
    blended = alpha * pred_val_nn + (1.0 - alpha) * pred_val_knn
    rmse = float(np.sqrt(mean_squared_error(y_val, blended)))
    if rmse < best_blend_rmse:
        best_blend_rmse = rmse
        best_alpha = float(alpha)

pred_test_blend = best_alpha * pred_test_nn + (1.0 - best_alpha) * pred_test_knn

rmse_nn = float(np.sqrt(mean_squared_error(y_test, pred_test_nn)))
rmse_knn = float(np.sqrt(mean_squared_error(y_test, pred_test_knn)))
rmse_blend = float(np.sqrt(mean_squared_error(y_test, pred_test_blend)))
r2_blend = r2_score(y_test, pred_test_blend)

# Calibrate final score to combine fit signal + player quality prior.
calib_A_val = np.column_stack([pred_val_nn * best_alpha + pred_val_knn * (1.0 - best_alpha), q_val, np.ones_like(q_val)])
calib_coef, _, _, _ = np.linalg.lstsq(calib_A_val, y_val, rcond=None)

calib_A_test = np.column_stack([pred_test_blend, q_test, np.ones_like(q_test)])
pred_test_calibrated = calib_A_test @ calib_coef
rmse_calibrated = float(np.sqrt(mean_squared_error(y_test, pred_test_calibrated)))
r2_calibrated = r2_score(y_test, pred_test_calibrated)

print('\n--- EVALUATION (Target: Marginal Net Points / 48) ---')
print(f'NN RMSE:    {rmse_nn:.4f}')
print(f'KNN RMSE:   {rmse_knn:.4f}')
print(f'Blend RMSE: {rmse_blend:.4f}  (alpha={best_alpha:.2f})')
print(f'Blend R2:   {r2_blend:.4f}')
print(
    f'Calibrated RMSE: {rmse_calibrated:.4f}  '
    f'(w_pred={calib_coef[0]:.3f}, w_quality={calib_coef[1]:.3f}, bias={calib_coef[2]:.3f})'
)
print(f'Calibrated R2:   {r2_calibrated:.4f}')

# -------------------------
# SAVE ARTIFACTS
# -------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

torch.save(
    {
        'state_dict': model.state_dict(),
        'input_dim': int(X_train.shape[1]),
        'model_version': 'context_aware_v2',
    },
    MODEL_PATH,
)

with open(ARTIFACT_PATH, 'wb') as f:
    pickle.dump(
        {
            'tracking_scaler': tracking_scaler,
            'input_scaler': input_scaler,
            'archetype_ids': archetype_ids,
            'player_feature_cols': player_feature_cols,
            'blend_alpha': best_alpha,
            'knn_model': knn,
            'score_calibration_coef': calib_coef.astype(np.float32),
            'player_quality_z': player_quality_z,
            'quality_col': 'QUALITY_PRIOR_Z',
            'global_train_mean_net48': global_train_mean,
            'core_prior_strength': CORE_PRIOR_STRENGTH,
            'model_version': 'context_aware_v2',
        },
        f,
    )

print(f'\nSaved neural weights to: {MODEL_PATH}')
print(f'Saved inference artifacts to: {ARTIFACT_PATH}')
print('Training complete.')
