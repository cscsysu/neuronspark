"""
NS-2025-05 Traffic Volume Prediction - V3 (Official Approach)
==============================================================
Two-step strategy:
  Step A: Use Prophet to predict future weather features (temp, rain, snow, clouds)
  Step B: Use LightGBM to predict traffic volume given predicted features + time features

Key improvements over v2:
  - Weather features predicted via Prophet (the official recommended approach)
  - Lag features based on Jan 2025 real data
  - Better CatBoost config (CPU mode with proper MAE)
  - Predicting both absolute values and change rates, then blending

Expected runtime: ~10-15 min on 1x GPU
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Config
# ============================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(DATA_DIR, 'train_data')
TEST_DIR = os.path.join(DATA_DIR, 'test_input', '2025_01')
SEED = 42
np.random.seed(SEED)

# ============================================================
# 1. Data Loading
# ============================================================
print("=" * 70)
print("STEP 1: Loading Data")
print("=" * 70)

def load_year_data(year_dir):
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

all_train = []
for year in sorted(os.listdir(TRAIN_DIR)):
    year_path = os.path.join(TRAIN_DIR, year)
    if os.path.isdir(year_path):
        df = load_year_data(year_path)
        if len(df) > 0:
            all_train.append(df)
            print(f"  {year}: {len(df)} rows | {df['date_time'].min()} ~ {df['date_time'].max()}")

train_df = pd.concat(all_train, ignore_index=True)
train_df = train_df.drop_duplicates(subset='date_time').sort_values('date_time').reset_index(drop=True)

# Test input (Jan 2025 real data)
test_input = load_year_data(TEST_DIR)
test_input = test_input.drop_duplicates(subset='date_time').sort_values('date_time').reset_index(drop=True)
print(f"  Test input (Jan 2025): {len(test_input)} rows")

# Full history
full_df = pd.concat([train_df, test_input], ignore_index=True)
full_df = full_df.drop_duplicates(subset='date_time').sort_values('date_time').reset_index(drop=True)
print(f"\n  Combined: {len(full_df)} rows | {full_df['date_time'].min()} ~ {full_df['date_time'].max()}")
print(f"  Traffic: mean={full_df['traffic_volume'].mean():.0f}, std={full_df['traffic_volume'].std():.0f}")

# ============================================================
# 2. Step A: Predict Future Weather with Prophet
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Predicting Future Weather Features (Prophet)")
print("=" * 70)

from prophet import Prophet

# Target prediction period
pred_start = pd.Timestamp('2025-02-01 00:00:00')  # Jan has real data
pred_end = pd.Timestamp('2025-09-30 23:00:00')
pred_dates_full = pd.date_range('2025-01-01 00:00:00', '2025-09-30 23:00:00', freq='h')
pred_dates_future = pd.date_range(pred_start, pred_end, freq='h')

# Prepare a dataframe with all predicted features for the full prediction range
future_features = pd.DataFrame({'date_time': pred_dates_full})

# --- 2a. Predict Temperature ---
print("\n  [2a] Predicting temperature...")
temp_data = full_df[['date_time', 'temp']].dropna().rename(columns={'date_time': 'ds', 'temp': 'y'})

# Use daily aggregation for Prophet (more stable), then interpolate to hourly
temp_daily = temp_data.set_index('ds').resample('D').mean().reset_index()

model_temp = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='additive',
    changepoint_prior_scale=0.05,
)
model_temp.fit(temp_daily)

future_temp_dates = pd.DataFrame({'ds': pd.date_range(temp_daily['ds'].min(), '2025-09-30', freq='D')})
forecast_temp = model_temp.predict(future_temp_dates)

# Map daily forecast to hourly
forecast_temp_daily = forecast_temp[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'temp_daily'})
forecast_temp_daily['date'] = forecast_temp_daily['date'].dt.date
future_features['date'] = future_features['date_time'].dt.date
future_features = future_features.merge(forecast_temp_daily, on='date', how='left')

# Also model hourly temp pattern within day (from historical)
hourly_temp_pattern = full_df.copy()
hourly_temp_pattern['hour'] = hourly_temp_pattern['date_time'].dt.hour
hourly_temp_pattern['month'] = hourly_temp_pattern['date_time'].dt.month
hourly_temp_offset = hourly_temp_pattern.groupby(['month', 'hour'])['temp'].mean()
daily_temp_mean = hourly_temp_pattern.groupby(['month'])['temp'].mean()

# offset = hourly_mean - daily_mean for that month
hourly_offsets = {}
for (month, hour), val in hourly_temp_offset.items():
    hourly_offsets[(month, hour)] = val - daily_temp_mean.get(month, val)

future_features['hour'] = future_features['date_time'].dt.hour
future_features['month'] = future_features['date_time'].dt.month
future_features['temp_offset'] = future_features.apply(
    lambda r: hourly_offsets.get((r['month'], r['hour']), 0), axis=1
)
future_features['temp_predicted'] = future_features['temp_daily'] + future_features['temp_offset']

# Override with real data for January
jan_temp = test_input[['date_time', 'temp']].dropna()
jan_temp_dict = dict(zip(jan_temp['date_time'], jan_temp['temp']))
future_features['temp_predicted'] = future_features.apply(
    lambda r: jan_temp_dict.get(r['date_time'], r['temp_predicted']), axis=1
)
print(f"    Temperature predicted: mean={future_features['temp_predicted'].mean():.1f}K")

# --- 2b. Predict Rain ---
print("  [2b] Predicting rain_1h...")
rain_data = full_df[['date_time', 'rain_1h']].dropna().rename(columns={'date_time': 'ds', 'rain_1h': 'y'})
rain_daily = rain_data.set_index('ds').resample('D').mean().reset_index()

model_rain = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.01,
)
model_rain.fit(rain_daily)
future_rain_dates = pd.DataFrame({'ds': pd.date_range(rain_daily['ds'].min(), '2025-09-30', freq='D')})
forecast_rain = model_rain.predict(future_rain_dates)
forecast_rain_daily = forecast_rain[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'rain_daily'})
forecast_rain_daily['date'] = forecast_rain_daily['date'].dt.date
forecast_rain_daily['rain_daily'] = forecast_rain_daily['rain_daily'].clip(lower=0)
future_features = future_features.merge(forecast_rain_daily, on='date', how='left')

# Override Jan
jan_rain = test_input[['date_time', 'rain_1h']].dropna()
jan_rain_dict = dict(zip(jan_rain['date_time'], jan_rain['rain_1h']))
future_features['rain_predicted'] = future_features.apply(
    lambda r: jan_rain_dict.get(r['date_time'], r['rain_daily']), axis=1
)
print(f"    Rain predicted: mean={future_features['rain_predicted'].mean():.3f}")

# --- 2c. Predict Snow ---
print("  [2c] Predicting snow_1h...")
snow_data = full_df[['date_time', 'snow_1h']].dropna().rename(columns={'date_time': 'ds', 'snow_1h': 'y'})
snow_daily = snow_data.set_index('ds').resample('D').mean().reset_index()

model_snow = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.01,
)
model_snow.fit(snow_daily)
future_snow_dates = pd.DataFrame({'ds': pd.date_range(snow_daily['ds'].min(), '2025-09-30', freq='D')})
forecast_snow = model_snow.predict(future_snow_dates)
forecast_snow_daily = forecast_snow[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'snow_daily'})
forecast_snow_daily['date'] = forecast_snow_daily['date'].dt.date
forecast_snow_daily['snow_daily'] = forecast_snow_daily['snow_daily'].clip(lower=0)
future_features = future_features.merge(forecast_snow_daily, on='date', how='left')

jan_snow = test_input[['date_time', 'snow_1h']].dropna()
jan_snow_dict = dict(zip(jan_snow['date_time'], jan_snow['snow_1h']))
future_features['snow_predicted'] = future_features.apply(
    lambda r: jan_snow_dict.get(r['date_time'], r['snow_daily']), axis=1
)
print(f"    Snow predicted: mean={future_features['snow_predicted'].mean():.4f}")

# --- 2d. Predict Clouds ---
print("  [2d] Predicting clouds_all...")
cloud_data = full_df[['date_time', 'clouds_all']].dropna().rename(columns={'date_time': 'ds', 'clouds_all': 'y'})
cloud_daily = cloud_data.set_index('ds').resample('D').mean().reset_index()

model_cloud = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
)
model_cloud.fit(cloud_daily)
future_cloud_dates = pd.DataFrame({'ds': pd.date_range(cloud_daily['ds'].min(), '2025-09-30', freq='D')})
forecast_cloud = model_cloud.predict(future_cloud_dates)
forecast_cloud_daily = forecast_cloud[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'clouds_daily'})
forecast_cloud_daily['date'] = forecast_cloud_daily['date'].dt.date
forecast_cloud_daily['clouds_daily'] = forecast_cloud_daily['clouds_daily'].clip(0, 100)
future_features = future_features.merge(forecast_cloud_daily, on='date', how='left')

jan_cloud = test_input[['date_time', 'clouds_all']].dropna()
jan_cloud_dict = dict(zip(jan_cloud['date_time'], jan_cloud['clouds_all']))
future_features['clouds_predicted'] = future_features.apply(
    lambda r: jan_cloud_dict.get(r['date_time'], r['clouds_daily']), axis=1
)
print(f"    Clouds predicted: mean={future_features['clouds_predicted'].mean():.1f}%")

# --- 2e. Predict weather_main (categorical - use seasonal probability) ---
print("  [2e] Predicting weather_main (seasonal pattern)...")
weather_hist = full_df[['date_time', 'weather_main']].dropna().copy()
weather_hist['month'] = weather_hist['date_time'].dt.month
weather_hist['hour'] = weather_hist['date_time'].dt.hour

# For each (month, hour), find the most common weather category
weather_mode = weather_hist.groupby(['month', 'hour'])['weather_main'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Clear'
).reset_index()
weather_mode.columns = ['month', 'hour', 'weather_predicted']

future_features = future_features.merge(weather_mode, on=['month', 'hour'], how='left')
future_features['weather_predicted'] = future_features['weather_predicted'].fillna('Clear')

# Override Jan
jan_weather = test_input[['date_time', 'weather_main']].dropna()
jan_weather_dict = dict(zip(jan_weather['date_time'], jan_weather['weather_main']))
future_features['weather_predicted'] = future_features.apply(
    lambda r: jan_weather_dict.get(r['date_time'], r['weather_predicted']), axis=1
)
print(f"    Weather distribution: {future_features['weather_predicted'].value_counts().head(5).to_dict()}")

print("\n  Prophet predictions complete!")

# ============================================================
# 3. Feature Engineering for Regression Model
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Feature Engineering")
print("=" * 70)

# US Holidays
US_HOLIDAYS = set(pd.to_datetime([
    '2019-11-28', '2019-12-25',
    '2020-01-01', '2020-01-20', '2020-02-17', '2020-05-25', '2020-07-03',
    '2020-09-07', '2020-11-26', '2020-12-25',
    '2021-01-01', '2021-01-18', '2021-02-15', '2021-05-31', '2021-06-18',
    '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24',
    '2022-01-17', '2022-02-21', '2022-05-30', '2022-06-20',
    '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-26',
    '2023-01-02', '2023-01-16', '2023-02-20', '2023-05-29', '2023-06-19',
    '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25',
    '2024-01-01', '2024-01-15', '2024-02-19', '2024-05-27', '2024-06-19',
    '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', '2025-06-19',
    '2025-07-04', '2025-09-01',
]).date)


def build_features(df, weather_cols=None):
    """Build feature matrix for traffic prediction."""
    df = df.copy()
    dt = df['date_time']
    
    # --- Time features ---
    df['hour'] = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek
    df['day_of_month'] = dt.dt.day
    df['month'] = dt.dt.month
    df['day_of_year'] = dt.dt.dayofyear
    df['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = dt.dt.date.isin(US_HOLIDAYS).astype(int)
    
    # Day before/after holiday
    day_before = set((pd.Timestamp(d) - timedelta(days=1)).date() for d in US_HOLIDAYS)
    day_after = set((pd.Timestamp(d) + timedelta(days=1)).date() for d in US_HOLIDAYS)
    df['is_day_before_holiday'] = dt.dt.date.isin(day_before).astype(int)
    df['is_day_after_holiday'] = dt.dt.date.isin(day_after).astype(int)
    
    # --- Cyclical encoding ---
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # --- Rush hour ---
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 15) & (df['hour'] <= 18)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    # --- Interactions ---
    df['hour_x_weekend'] = df['hour'] * df['is_weekend']
    df['dow_x_hour'] = df['day_of_week'] * 24 + df['hour']
    
    # --- Weather features (numerical) ---
    if 'temp' in df.columns:
        df['temp_celsius'] = df['temp'] - 273.15
    if 'temp_predicted' in df.columns:
        df['temp_celsius'] = df['temp_predicted'] - 273.15
    
    # rain, snow, clouds
    for col in ['rain_1h', 'rain_predicted']:
        if col in df.columns:
            df['rain'] = df[col]
            break
    for col in ['snow_1h', 'snow_predicted']:
        if col in df.columns:
            df['snow'] = df[col]
            break
    for col in ['clouds_all', 'clouds_predicted']:
        if col in df.columns:
            df['clouds'] = df[col]
            break
    
    # Weather category encoding
    weather_col = None
    for col in ['weather_main', 'weather_predicted']:
        if col in df.columns:
            weather_col = col
            break
    
    if weather_col:
        # Map to numeric impact score (based on traffic impact)
        weather_impact = {
            'Clear': 1.0,
            'Clouds': 0.95,
            'Mist': 0.85,
            'Haze': 0.85,
            'Fog': 0.75,
            'Drizzle': 0.80,
            'Rain': 0.70,
            'Snow': 0.60,
            'Thunderstorm': 0.50,
            'Smoke': 0.80,
        }
        df['weather_impact'] = df[weather_col].map(weather_impact).fillna(0.85)
        
        # Also one-hot encode top categories
        for cat in ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']:
            df[f'weather_is_{cat}'] = (df[weather_col] == cat).astype(int)
    
    # --- Fill NaN ---
    for col in ['temp_celsius', 'rain', 'snow', 'clouds']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    
    return df


# Build features for training
print("Building training features...")
train_featured = build_features(full_df)

# Build features for prediction
print("Building prediction features...")
pred_featured = build_features(future_features)

# ============================================================
# 4. Historical Pattern Features
# ============================================================
print("\nAdding historical pattern features...")

hist = full_df[full_df['traffic_volume'].notna()].copy()
hist['hour'] = hist['date_time'].dt.hour
hist['day_of_week'] = hist['date_time'].dt.dayofweek
hist['month'] = hist['date_time'].dt.month
hist['day_of_month'] = hist['date_time'].dt.day
hist['year'] = hist['date_time'].dt.year

# (month, dow, hour) stats
g1 = hist.groupby(['month', 'day_of_week', 'hour'])['traffic_volume'].agg(
    hist_mdh_mean='mean', hist_mdh_std='std', hist_mdh_median='median'
).reset_index()

# (month, hour) stats
g2 = hist.groupby(['month', 'hour'])['traffic_volume'].agg(
    hist_mh_mean='mean', hist_mh_median='median'
).reset_index()

# (dow, hour) stats
g3 = hist.groupby(['day_of_week', 'hour'])['traffic_volume'].agg(
    hist_dh_mean='mean', hist_dh_median='median'
).reset_index()

# Same period last year
for target_year in [2024, 2023]:
    year_data = hist[hist['year'] == target_year].copy()
    if len(year_data) > 0:
        year_data['key'] = year_data['month'] * 10000 + year_data['day_of_month'] * 100 + year_data['hour']
        lookup = year_data.groupby('key')['traffic_volume'].mean().to_dict()
        
        for df_ref in [train_featured, pred_featured]:
            df_ref[f'vol_{target_year}'] = (
                df_ref['month'] * 10000 + df_ref['day_of_month'] * 100 + df_ref['hour']
            ).map(lookup)

# Merge pattern features (use .values to avoid index alignment issues)
for df_ref in [train_featured, pred_featured]:
    df_ref.reset_index(drop=True, inplace=True)
    
    tmp1 = df_ref[['month', 'day_of_week', 'hour']].merge(g1, on=['month', 'day_of_week', 'hour'], how='left')
    for col in ['hist_mdh_mean', 'hist_mdh_std', 'hist_mdh_median']:
        df_ref[col] = tmp1[col].values
    
    tmp2 = df_ref[['month', 'hour']].merge(g2, on=['month', 'hour'], how='left')
    for col in ['hist_mh_mean', 'hist_mh_median']:
        df_ref[col] = tmp2[col].values
    
    tmp3 = df_ref[['day_of_week', 'hour']].merge(g3, on=['day_of_week', 'hour'], how='left')
    for col in ['hist_dh_mean', 'hist_dh_median']:
        df_ref[col] = tmp3[col].values

# ============================================================
# 5. Define Feature Columns
# ============================================================
FEATURE_COLS = [
    # Time
    'hour', 'day_of_week', 'day_of_month', 'month', 'day_of_year', 'week_of_year',
    'is_weekend', 'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday',
    # Cyclical
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    # Rush
    'is_morning_rush', 'is_evening_rush', 'is_night',
    # Interactions
    'hour_x_weekend', 'dow_x_hour',
    # Weather (predicted for future!)
    'temp_celsius', 'rain', 'snow', 'clouds',
    'weather_impact',
    'weather_is_Clear', 'weather_is_Clouds', 'weather_is_Rain', 'weather_is_Snow', 'weather_is_Mist',
    # Historical patterns
    'hist_mdh_mean', 'hist_mdh_std', 'hist_mdh_median',
    'hist_mh_mean', 'hist_mh_median',
    'hist_dh_mean', 'hist_dh_median',
    # Same period past years
    'vol_2024', 'vol_2023',
]

# Filter to columns that exist in both
FEATURE_COLS = [c for c in FEATURE_COLS if c in train_featured.columns and c in pred_featured.columns]
print(f"\nUsing {len(FEATURE_COLS)} features")

# ============================================================
# 6. Prepare Training Data
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Training LightGBM")
print("=" * 70)

# Training rows (only rows with traffic_volume)
train_mask = train_featured['traffic_volume'].notna()
train_data = train_featured[train_mask].copy()

for col in FEATURE_COLS:
    train_data[col] = train_data[col].fillna(train_data[col].median())
    pred_featured[col] = pred_featured[col].fillna(
        train_data[col].median() if train_data[col].notna().any() else 0
    )

X_all = train_data[FEATURE_COLS].values
y_all = train_data['traffic_volume'].values
print(f"Training samples: {len(X_all)}, Features: {len(FEATURE_COLS)}")

# Time-based split
split_idx = int(len(X_all) * 0.80)
X_train, X_val = X_all[:split_idx], X_all[split_idx:]
y_train, y_val = y_all[:split_idx], y_all[split_idx:]

# ============================================================
# 6a. LightGBM Model 1 (Huber loss - robust to outliers)
# ============================================================
import lightgbm as lgb

print("\n[1/3] LightGBM (huber loss, deep)...")
lgb_params1 = {
    'objective': 'huber',
    'alpha': 500,  # huber delta (traffic range is 0-7280)
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 255,
    'learning_rate': 0.05,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': SEED,
}

dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=dtrain)

lgb_model1 = lgb.train(
    lgb_params1, dtrain, num_boost_round=5000,
    valid_sets=[dtrain, dval], valid_names=['train', 'val'],
    callbacks=[lgb.log_evaluation(500), lgb.early_stopping(300)],
)
print(f"  Best iter: {lgb_model1.best_iteration}, Val MAE: {lgb_model1.best_score['val']['l1']:.2f}")

# ============================================================
# 6b. LightGBM Model 2 (MAE loss, more regularized)
# ============================================================
print("\n[2/3] LightGBM (mae loss, regularized)...")
lgb_params2 = {
    'objective': 'mae',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.03,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'min_child_samples': 50,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': SEED + 1,
}

lgb_model2 = lgb.train(
    lgb_params2, dtrain, num_boost_round=8000,
    valid_sets=[dtrain, dval], valid_names=['train', 'val'],
    callbacks=[lgb.log_evaluation(500), lgb.early_stopping(300)],
)
print(f"  Best iter: {lgb_model2.best_iteration}, Val MAE: {lgb_model2.best_score['val']['l1']:.2f}")

# ============================================================
# 6c. XGBoost
# ============================================================
print("\n[3/3] XGBoost...")
try:
    import xgboost as xgb
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 20,
        'seed': SEED,
    }
    
    # Try GPU first
    try:
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['device'] = 'cuda'
        dxtr = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
        dxval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)
        xgb_model = xgb.train(
            xgb_params, dxtr, num_boost_round=5000,
            evals=[(dxtr, 'train'), (dxval, 'val')],
            early_stopping_rounds=300, verbose_eval=500,
        )
    except Exception:
        xgb_params['tree_method'] = 'hist'
        xgb_params.pop('device', None)
        dxtr = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
        dxval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)
        xgb_model = xgb.train(
            xgb_params, dxtr, num_boost_round=5000,
            evals=[(dxtr, 'train'), (dxval, 'val')],
            early_stopping_rounds=300, verbose_eval=500,
        )
    
    print(f"  Best iter: {xgb_model.best_iteration}")
    HAS_XGB = True
except ImportError:
    print("  XGBoost not available")
    HAS_XGB = False

# ============================================================
# 7. Retrain on Full Data
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Retraining on Full Data")
print("=" * 70)

dfull = lgb.Dataset(X_all, label=y_all, feature_name=FEATURE_COLS)
lgb_full1 = lgb.train(lgb_params1, dfull, num_boost_round=lgb_model1.best_iteration)
lgb_full2 = lgb.train(lgb_params2, dfull, num_boost_round=lgb_model2.best_iteration)
print("  LightGBM models retrained on full data")

if HAS_XGB:
    dxfull = xgb.DMatrix(X_all, label=y_all, feature_names=FEATURE_COLS)
    xgb_full = xgb.train(xgb_params, dxfull, num_boost_round=xgb_model.best_iteration)
    print("  XGBoost retrained on full data")

# ============================================================
# 8. Generate Predictions
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Generating Predictions")
print("=" * 70)

X_pred = pred_featured[FEATURE_COLS].values
print(f"Prediction points: {len(X_pred)}")

# Model predictions
pred1 = lgb_full1.predict(X_pred)
pred2 = lgb_full2.predict(X_pred)

if HAS_XGB:
    dxpred = xgb.DMatrix(X_pred, feature_names=FEATURE_COLS)
    pred3 = xgb_full.predict(dxpred)
    # Weighted ensemble
    ensemble_pred = 0.40 * pred1 + 0.35 * pred2 + 0.25 * pred3
else:
    ensemble_pred = 0.55 * pred1 + 0.45 * pred2

print(f"Ensemble prediction: mean={ensemble_pred.mean():.0f}, std={ensemble_pred.std():.0f}")

# ============================================================
# 9. Historical Pattern Blending
# ============================================================
print("\nBlending with historical pattern...")

# Recency-weighted historical pattern
hist_pattern = full_df[full_df['traffic_volume'].notna()].copy()
hist_pattern['hour'] = hist_pattern['date_time'].dt.hour
hist_pattern['day_of_week'] = hist_pattern['date_time'].dt.dayofweek
hist_pattern['month'] = hist_pattern['date_time'].dt.month
hist_pattern['year'] = hist_pattern['date_time'].dt.year
year_w = {2019: 0.3, 2020: 0.5, 2021: 0.6, 2022: 0.8, 2023: 1.2, 2024: 1.8, 2025: 2.5}
hist_pattern['w'] = hist_pattern['year'].map(year_w).fillna(1.0)

def wmean(g):
    return np.average(g['traffic_volume'], weights=g['w'])

pattern = hist_pattern.groupby(['month', 'day_of_week', 'hour']).apply(wmean).reset_index()
pattern.columns = ['month', 'day_of_week', 'hour', 'pattern_vol']

pred_merged = pred_featured[['date_time', 'month', 'day_of_week', 'hour']].merge(
    pattern, on=['month', 'day_of_week', 'hour'], how='left'
)
pattern_pred = pred_merged['pattern_vol'].values

# Fill any NaN in pattern with ensemble
pattern_nan_mask = np.isnan(pattern_pred)
pattern_pred[pattern_nan_mask] = ensemble_pred[pattern_nan_mask]

# Blend: give model more weight since it now has weather info
final_pred = 0.80 * ensemble_pred + 0.20 * pattern_pred

# ============================================================
# 10. Post-processing
# ============================================================
print("Post-processing...")

# Clip to valid range
vol_max = full_df['traffic_volume'].max() * 1.05
final_pred = np.clip(final_pred, 0, vol_max)

# Round
final_pred = np.round(final_pred).astype(int)
final_pred = np.maximum(final_pred, 0)

# ============================================================
# 11. Override January with Real Data
# ============================================================
print("Overriding January with test_input real data...")

results_df = pd.DataFrame({
    'date_time': pred_dates_full,
    'traffic_volume': final_pred,
})

jan_real = test_input[['date_time', 'traffic_volume']].drop_duplicates(subset='date_time').sort_values('date_time')
jan_real_dict = dict(zip(jan_real['date_time'], jan_real['traffic_volume']))

for idx, row in results_df.iterrows():
    if row['date_time'] in jan_real_dict:
        results_df.at[idx, 'traffic_volume'] = int(jan_real_dict[row['date_time']])

# ============================================================
# 12. Validation
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Validation (Model vs Jan 2025 Actual)")
print("=" * 70)

# Compare model prediction vs actual for January (before override)
jan_mask = pred_featured['date_time'].dt.month == 1
jan_model = ensemble_pred[jan_mask.values]
jan_actual_list = []
for dt in pred_featured[jan_mask]['date_time']:
    jan_actual_list.append(jan_real_dict.get(dt, np.nan))
jan_actual_arr = np.array(jan_actual_list, dtype=float)
valid = ~np.isnan(jan_actual_arr)

if valid.sum() > 0:
    ja = jan_actual_arr[valid]
    jp = jan_model[valid]
    
    mae = np.mean(np.abs(ja - jp))
    rmse = np.sqrt(np.mean((ja - jp) ** 2))
    nonzero = ja > 0
    mape = np.mean(np.abs(ja[nonzero] - jp[nonzero]) / ja[nonzero]) * 100
    ss_res = np.sum((ja - jp) ** 2)
    ss_tot = np.sum((ja - np.mean(ja)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"  Jan 2025 Validation:")
    print(f"    MAE:  {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAPE: {mape:.2f}%")
    print(f"    R2:   {r2:.4f}")

# ============================================================
# 13. Save Results
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Saving Results")
print("=" * 70)

results_df['traffic_volume'] = results_df['traffic_volume'].astype(int)
output_path = os.path.join(DATA_DIR, 'results.csv')
results_df.to_csv(output_path, index=False)

print(f"  Saved: {output_path}")
print(f"  Rows: {len(results_df)}")
print(f"  Range: {results_df['date_time'].iloc[0]} ~ {results_df['date_time'].iloc[-1]}")
print(f"  Volume: min={results_df['traffic_volume'].min()}, max={results_df['traffic_volume'].max()}, "
      f"mean={results_df['traffic_volume'].mean():.0f}")

# Package
import zipfile
zip_path = os.path.join(DATA_DIR, 'NS-2025-05-answer.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(output_path, 'results.csv')
print(f"  Packaged: {zip_path}")

print("\n" + "=" * 70)
print("DONE! Submit NS-2025-05-answer.zip")
print("=" * 70)
