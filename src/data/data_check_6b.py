import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# --- Load and merge data (done once for both scripts) ---
tier1 = pd.read_parquet('data/transition1x/descriptors/stream_b_geometric_tier1.parquet')
tier2 = pd.read_parquet('data/transition1x/descriptors/stream_b_geometric_tier2_raw.parquet')
df = tier1.merge(tier2, on='rxn_id', how='inner')
valid = df[df['E_NEB_strain'].notna() & df['E_strain_IDPP'].notna()].copy()

# ==========================================
# PART 1: Stats and Correlation Analysis
# ==========================================
print(f'Reactions with both values: {len(valid)}\n')

print('E_strain_IDPP stats:')
print(valid['E_strain_IDPP'].describe())
print()

print('E_NEB_strain stats:')
print(valid['E_NEB_strain'].describe())
print()

# Raw Pearson correlation
corr = valid['E_strain_IDPP'].corr(valid['E_NEB_strain'])
print(f'Pearson correlation: {corr:.4f}')
print(f'R² from correlation: {corr**2:.4f}\n')

# Check by neb_quality
print('Pearson r by neb_quality:')
for q in valid['neb_quality'].unique():
    sub = valid[valid['neb_quality'] == q]
    r = sub['E_strain_IDPP'].corr(sub['E_NEB_strain'])
    print(f'  {q}: r={r:.4f}, R²={r**2:.4f}, n={len(sub)}')
print()

# Check sign agreement
same_sign = ((valid['E_strain_IDPP'] > valid['E_strain_IDPP'].median()) == 
             (valid['E_NEB_strain'] > valid['E_NEB_strain'].median())).mean()
print(f'Rank agreement (both above/below median): {same_sign:.3f}')

# Check if there is a systematic offset
diff = valid['E_NEB_strain'] - valid['E_strain_IDPP']
print(f'\nSystematic offset (E_NEB - E_IDPP):')
print(f'  mean: {diff.mean():.4f} eV')
print(f'  std:  {diff.std():.4f} eV')
print(f'  E_NEB consistently higher: {(diff > 0).mean():.3f} of cases')

print('\n' + '-'*40 + '\n')

# ==========================================
# PART 2: Predictive Modeling Diagnostics
# ==========================================
X = valid[['E_strain_IDPP']].values
y = valid['E_NEB_strain'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear — if the relationship exists at all it should show here
lr = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f'Linear  R²: {r2_score(y_test, y_pred_lr):.4f}  MAE: {mean_absolute_error(y_test, y_pred_lr):.4f} eV')
print(f'  slope={lr.coef_[0]:.4f}  intercept={lr.intercept_:.4f}\n')

# RF — captures non-linear relationship if it exists
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f'RF      R²: {r2_score(y_test, y_pred_rf):.4f}  MAE: {mean_absolute_error(y_test, y_pred_rf):.4f} eV')