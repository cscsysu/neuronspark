"""
NS-2025-05 Traffic Volume Prediction Solution
Predict hourly traffic volume for 2025-01-01 to 2025-09-30
Uses ensemble of LightGBM + historical pattern matching
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 1. Data Loading
# ============================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(DATA_DIR, 'train_data')
TEST_DIR = os.path.join(DATA_DIR, 'test_input', '2025_01')

def load_year_data(year_dir):
    """Load all CSVs for a given year and merge on date_time."""
    files = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main']
    dfs = []
    for f in files:
        path = os.path.join(year_dir, f'{f}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['date_time'])
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on='date_time', how='outer')
    return merged

print("Loading training data...")
all_train = []
for year in sorted(os.listdir(TRAIN_DIR)):
    year_path = os.path.join(TRAIN_DIR, year)
    if os.path.isdir(year_path):
        df = load_year_data(year_path)
        if len(df) > 0:
            all_train.append(df)
            print(f"  {year}: {len(df)} rows, {df['date_time'].min()} ~ {df['date_time'].max()}")

train_df = pd.concat(all_train, ignore_index=True)
train_df = train_df.drop_duplicates(subset='date_time').sort_values('date_time').reset_index(drop=True)

# Load test input (Jan 2025 warm-up data)
print("\nLoading test input (Jan 2025)...")
test_input = load_year_data(TEST_DIR)
test_input = test_input.drop_duplicates(subset='date_time').sort_values('date_time').reset_index(drop=True)
print(f"  Test input: {len(test_input)} rows, {test_input['date_time'].min()} ~ {test_input['date_time'].max()}")

# Combine all available data
full_df = pd.concat([train_df, test_input], ignore_index=True)
full_df = full_df.drop_duplicates(subset='date_time').sort_values('date_time').reset_index(drop=True)
print(f"\nTotal data: {len(full_df)} rows")
print(f"Date range: {full_df['date_time'].min()} ~ {full_df['date_time'].max()}")
print(f"Traffic volume stats:\n{full_df['traffic_volume'].describe()}")

# ============================================================
# 2. Feature Engineering
# ============================================================
def create_time_features(df):
    """Create temporal features from date_time."""
    df = df.copy()
    dt = df['date_time']
    df['hour'] = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek  # 0=Monday
    df['day_of_month'] = dt.dt.day
    df['month'] = dt.dt.month
    df['day_of_year'] = dt.dt.dayofyear
    df['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for periodic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # US public holidays (approximate, this is likely a US city based on the data pattern)
    us_holidays_2025 = [
        '2025-01-01',  # New Year
        '2025-01-20',  # MLK Day
        '2025-02-17',  # Presidents Day
        '2025-05-26',  # Memorial Day
        '2025-06-19',  # Juneteenth
        '2025-07-04',  # Independence Day
        '2025-09-01',  # Labor Day
    ]
    holiday_dates = pd.to_datetime(us_holidays_2025)
    df['is_holiday'] = dt.dt.date.isin(holiday_dates.date).astype(int)
    
    # Rush hour indicators
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    return df

def create_historical_features(df, full_history):
    """Create features based on historical same-period patterns."""
    df = df.copy()
    
    # Build lookup: for each (month, day, hour, day_of_week), compute historical stats
    history = full_history.copy()
    history['hour'] = history['date_time'].dt.hour
    history['day_of_week'] = history['date_time'].dt.dayofweek
    history['month'] = history['date_time'].dt.month
    history['day_of_month'] = history['date_time'].dt.day
    
    # Historical mean/std by (month, hour, day_of_week)
    hist_stats = history.groupby(['month', 'hour', 'day_of_week'])['traffic_volume'].agg(
        ['mean', 'std', 'median']
    ).reset_index()
    hist_stats.columns = ['month', 'hour', 'day_of_week', 'hist_mean', 'hist_std', 'hist_median']
    
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    
    df = df.merge(hist_stats, on=['month', 'hour', 'day_of_week'], how='left')
    
    # Historical mean by (month, hour) - more general
    hist_mh = history.groupby(['month', 'hour'])['traffic_volume'].agg(['mean', 'median']).reset_index()
    hist_mh.columns = ['month', 'hour', 'hist_month_hour_mean', 'hist_month_hour_median']
    df = df.merge(hist_mh, on=['month', 'hour'], how='left')
    
    # Historical mean by (day_of_week, hour) - weekly pattern
    hist_dh = history.groupby(['day_of_week', 'hour'])['traffic_volume'].agg(['mean', 'median']).reset_index()
    hist_dh.columns = ['day_of_week', 'hour', 'hist_dow_hour_mean', 'hist_dow_hour_median']
    df = df.merge(hist_dh, on=['day_of_week', 'hour'], how='left')
    
    # Same date last year (2024) traffic
    history_2024 = history[history['date_time'].dt.year == 2024].copy()
    history_2024['month_day_hour'] = (history_2024['month'] * 10000 + 
                                      history_2024['day_of_month'] * 100 + 
                                      history_2024['hour'])
    lookup_2024 = history_2024.groupby('month_day_hour')['traffic_volume'].mean().to_dict()
    
    df['day_of_month'] = df['date_time'].dt.day
    df['month_day_hour'] = df['month'] * 10000 + df['day_of_month'] * 100 + df['hour']
    df['last_year_volume'] = df['month_day_hour'].map(lookup_2024)
    
    # Same date 2023
    history_2023 = history[history['date_time'].dt.year == 2023].copy()
    history_2023['month_day_hour'] = (history_2023['date_time'].dt.month * 10000 + 
                                      history_2023['date_time'].dt.day * 100 + 
                                      history_2023['date_time'].dt.hour)
    lookup_2023 = history_2023.groupby('month_day_hour')['traffic_volume'].mean().to_dict()
    df['two_years_ago_volume'] = df['month_day_hour'].map(lookup_2023)
    
    # Average of last 2 years same period
    df['avg_last_2y'] = df[['last_year_volume', 'two_years_ago_volume']].mean(axis=1)
    
    # Drop temp columns
    df.drop(columns=['month_day_hour', 'day_of_month'], inplace=True, errors='ignore')
    
    return df

# ============================================================
# 3. Prepare Training Data
# ============================================================
print("\n" + "="*60)
print("Preparing features...")

# Create complete hourly index for training data
train_full = full_df[full_df['traffic_volume'].notna()].copy()

# Add time features
train_full = create_time_features(train_full)
train_full = create_historical_features(train_full, train_full)

# Encode weather_main
if 'weather_main' in train_full.columns:
    weather_dummies = pd.get_dummies(train_full['weather_main'], prefix='weather')
    train_full = pd.concat([train_full, weather_dummies], axis=1)

# Convert temp to Celsius for better interpretability
if 'temp' in train_full.columns:
    train_full['temp_celsius'] = train_full['temp'] - 273.15

# Features for the model
feature_cols = [
    'hour', 'day_of_week', 'month', 'day_of_year', 'week_of_year',
    'is_weekend', 'is_holiday', 'is_morning_rush', 'is_evening_rush', 'is_night',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'doy_sin', 'doy_cos',
    'hist_mean', 'hist_std', 'hist_median',
    'hist_month_hour_mean', 'hist_month_hour_median',
    'hist_dow_hour_mean', 'hist_dow_hour_median',
    'last_year_volume', 'two_years_ago_volume', 'avg_last_2y',
]

# Add weather features if available (only for training, not prediction)
weather_cols = [c for c in train_full.columns if c.startswith('weather_')]
# Don't include weather in features since we don't have it for prediction period

# Add temp if available
if 'temp_celsius' in train_full.columns:
    # We won't use temp for prediction since we don't have future weather
    pass

# Filter to valid rows
valid_mask = train_full['traffic_volume'].notna()
train_data = train_full[valid_mask].copy()

# Handle NaN in features
for col in feature_cols:
    if col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].median())

X_train = train_data[feature_cols].values
y_train = train_data['traffic_volume'].values

print(f"Training samples: {len(X_train)}")
print(f"Features: {len(feature_cols)}")

# ============================================================
# 4. Train LightGBM Model
# ============================================================
print("\n" + "="*60)
print("Training LightGBM model...")

# Time-based split for validation
split_idx = int(len(X_train) * 0.85)
X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_set)

params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

callbacks = [
    lgb.log_evaluation(200),
    lgb.early_stopping(100),
]

model = lgb.train(
    params,
    train_set,
    num_boost_round=3000,
    valid_sets=[train_set, val_set],
    valid_names=['train', 'valid'],
    callbacks=callbacks,
)

# Retrain on full data with best iteration
print(f"\nBest iteration: {model.best_iteration}")
print(f"Validation MAE: {mean_absolute_error(y_val, model.predict(X_val)):.2f}")

# Retrain on all data
print("\nRetraining on full dataset...")
full_train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
model_full = lgb.train(
    params,
    full_train_set,
    num_boost_round=model.best_iteration,
)

# Also train a second model with different params for ensemble
params2 = params.copy()
params2.update({
    'num_leaves': 63,
    'learning_rate': 0.03,
    'feature_fraction': 0.7,
    'min_child_samples': 50,
})

model2 = lgb.train(
    params2,
    train_set,
    num_boost_round=5000,
    valid_sets=[train_set, val_set],
    valid_names=['train', 'valid'],
    callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)],
)
print(f"Model2 best iteration: {model2.best_iteration}")

model2_full = lgb.train(
    params2,
    full_train_set,
    num_boost_round=model2.best_iteration,
)

# ============================================================
# 5. Generate Predictions
# ============================================================
print("\n" + "="*60)
print("Generating predictions for 2025-01-01 to 2025-09-30...")

# Create prediction dataframe
pred_dates = pd.date_range('2025-01-01 00:00:00', '2025-09-30 23:00:00', freq='h')
pred_df = pd.DataFrame({'date_time': pred_dates})
print(f"Prediction points: {len(pred_df)}")

# Add features
pred_df = create_time_features(pred_df)
pred_df = create_historical_features(pred_df, train_full)

# Handle NaN
for col in feature_cols:
    if col in pred_df.columns:
        pred_df[col] = pred_df[col].fillna(train_data[col].median() if col in train_data.columns else 0)

X_pred = pred_df[feature_cols].values

# Predict with both models
pred1 = model_full.predict(X_pred)
pred2 = model2_full.predict(X_pred)

# Ensemble: weighted average
lgb_pred = 0.6 * pred1 + 0.4 * pred2

# ============================================================
# 6. Historical Pattern Baseline
# ============================================================
print("Computing historical pattern baseline...")

# For each prediction point, use weighted average of same (month, day_of_week, hour) from history
# with more recent years weighted higher
history_for_pattern = full_df[full_df['traffic_volume'].notna()].copy()
history_for_pattern['hour'] = history_for_pattern['date_time'].dt.hour
history_for_pattern['day_of_week'] = history_for_pattern['date_time'].dt.dayofweek
history_for_pattern['month'] = history_for_pattern['date_time'].dt.month
history_for_pattern['year'] = history_for_pattern['date_time'].dt.year

# Weight by recency
year_weights = {2019: 0.5, 2020: 0.7, 2021: 0.8, 2022: 1.0, 2023: 1.2, 2024: 1.5, 2025: 2.0}
history_for_pattern['weight'] = history_for_pattern['year'].map(year_weights).fillna(1.0)

# Weighted mean for each (month, day_of_week, hour)
def weighted_mean(group):
    return np.average(group['traffic_volume'], weights=group['weight'])

pattern_baseline = history_for_pattern.groupby(['month', 'day_of_week', 'hour']).apply(
    weighted_mean
).reset_index()
pattern_baseline.columns = ['month', 'day_of_week', 'hour', 'pattern_pred']

pred_df_pattern = pred_df[['date_time', 'month', 'day_of_week', 'hour']].merge(
    pattern_baseline, on=['month', 'day_of_week', 'hour'], how='left'
)
pattern_pred = pred_df_pattern['pattern_pred'].fillna(pred_df['hist_mean']).values

# ============================================================
# 7. Final Ensemble
# ============================================================
print("Creating final ensemble...")

# Blend LightGBM and pattern baseline
# LightGBM is more sophisticated but pattern is robust for long-horizon
final_pred = 0.7 * lgb_pred + 0.3 * pattern_pred

# Clip to reasonable range
vol_min = 0
vol_max = full_df['traffic_volume'].quantile(0.999) * 1.1
final_pred = np.clip(final_pred, vol_min, vol_max)

# Round to integers (traffic volume is integer)
final_pred = np.round(final_pred).astype(int)

# ============================================================
# 8. Validate against test_input (Jan 2025)
# ============================================================
print("\n" + "="*60)
print("Validation against Jan 2025 actual data:")

jan_actual = test_input[['date_time', 'traffic_volume']].copy()
jan_actual = jan_actual.drop_duplicates(subset='date_time').sort_values('date_time')

# Match predictions for January
jan_pred_mask = pred_df['date_time'].isin(jan_actual['date_time'])
jan_pred_values = final_pred[jan_pred_mask.values]
jan_actual_matched = jan_actual[jan_actual['date_time'].isin(pred_df[jan_pred_mask]['date_time'])]
jan_actual_values = jan_actual_matched['traffic_volume'].values

if len(jan_actual_values) > 0 and len(jan_pred_values) > 0:
    min_len = min(len(jan_actual_values), len(jan_pred_values))
    jan_actual_values = jan_actual_values[:min_len]
    jan_pred_values = jan_pred_values[:min_len]
    
    mae = mean_absolute_error(jan_actual_values, jan_pred_values)
    rmse = np.sqrt(mean_squared_error(jan_actual_values, jan_pred_values))
    # MAPE (avoid division by zero)
    nonzero_mask = jan_actual_values > 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(jan_actual_values[nonzero_mask] - jan_pred_values[nonzero_mask]) / jan_actual_values[nonzero_mask]) * 100
    else:
        mape = float('inf')
    # R2
    ss_res = np.sum((jan_actual_values - jan_pred_values) ** 2)
    ss_tot = np.sum((jan_actual_values - np.mean(jan_actual_values)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R2:   {r2:.4f}")

# ============================================================
# 9. Save Results
# ============================================================
print("\n" + "="*60)
print("Saving results...")

results = pd.DataFrame({
    'date_time': pred_dates,
    'traffic_volume': final_pred,
})

output_path = os.path.join(DATA_DIR, 'results.csv')
results.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
print(f"Total rows: {len(results)}")
print(f"Date range: {results['date_time'].iloc[0]} ~ {results['date_time'].iloc[-1]}")
print(f"Volume range: {results['traffic_volume'].min()} ~ {results['traffic_volume'].max()}")
print(f"Volume mean: {results['traffic_volume'].mean():.0f}")

# Quick sanity check
print("\nSample predictions:")
print(results.head(10).to_string(index=False))
print("...")
print(results.tail(5).to_string(index=False))

print("\n✓ Done! Now zip with: zip NS-2025-05-answer.zip results.csv")
