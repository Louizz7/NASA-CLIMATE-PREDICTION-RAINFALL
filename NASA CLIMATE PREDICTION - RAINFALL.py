# ============================================================
# RAINFALL FORECASTING PROJECT
# Climate Forecasting using Machine Learning

# Author : Louis Mbagwu
# IDE    : Visual Studio Code
#
# Objective:
# 1. Forecast future Rainfall Patterns
#
# Dataset:
# NASA POWER Climate Risk Indices (1990–2024)
# ============================================================

#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

from sklearn.model_selection import (TimeSeriesSplit, RandomizedSearchCV, cross_val_score)
from sklearn.preprocessing import (OrdinalEncoder, StandardScaler)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error)
from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import loguniform
from xgboost import XGBRegressor
from pathlib import Path

import joblib
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)

pd.set_option("display.float_format", lambda x: f"{x:.3f}")


# DATA COLLECTION (LOAD DATASET)
CLIMATE_DATA = Path(
    r"C:\Users\USER\Documents\3MTT DATA SCIENCE\PROJECTS\CLIMATE DATA ANALYSIS - NASA\NASA CLIMATE PREDICTION - RAINFALL\nasa_power_climate_risk_indices_190_capitals_1990_2024.csv"
)

df = pd.read_csv(CLIMATE_DATA)
print("\nFirst Five Rows")
print(df.head())

print("\nDataset Shape")
rows, columns = df.shape
print(f"Number of Rows    : {rows:,}")
print(f"Number of Columns : {columns}")

print("\nColumn Names")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")

print("\nDataset Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe().T)

# CHECK FOR MISSING VALUES, DUPLICATE RECORDS, DATA TYPES, NUMERIC/CATEGORICAL COLUMN SUMMARY, INVALID VALUES

print("\nMISSING VALUES")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

print("\nDUPLICATE RECORDS")
duplicates = df.duplicated().sum()
print(f"Number of Duplicate Records : {duplicates}")

print("\nDATA TYPES")
print(df.dtypes.to_frame(name="Data Type"))

print("\nNUMERIC COLUMNS SUMMARY")
numeric_columns = df.select_dtypes(include=[np.number]).columns
print(df[numeric_columns].describe().T)

print("\nCATEGORICAL COLUMNS SUMMARY")
categorical_columns = df.select_dtypes(include=[object]).columns
print(df[categorical_columns].describe().T)

print("\nINVALID VALUE CHECKS") # These checks are based on physical limits of climate variables.
print(f"Negative Rainfall Records : {(df['precip_total_mm'] < 0).sum()}")
print(f"Humidity > 100%           : {(df['rh_mean_pct'] > 100).sum()}")
print(f"Humidity < 0%             : {(df['rh_mean_pct'] < 0).sum()}")
print(f"Negative Solar Radiation  : {(df['solar_total_mj'] < 0).sum()}")

#OUTLIER DETECTION USING IQR
print("\nOUTLIER DETECTION")
# Variables of greatest interest
outlier_columns = ["temp_mean_c", "temp_max_c", "temp_min_c", "precip_total_mm",
                   "rh_mean_pct", "wind_mean_ms"]

for col in outlier_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"{col:<20} : {len(outliers)} outliers")
    
# Visualize Outliers with BOXPLOTS

plt.figure(figsize=(15,8))

for i, column in enumerate(outlier_columns, start=1):
    plt.subplot(2,3,i)
    sns.boxplot(x=df[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# CHECK TIME SERIES COMPLETENESS
print("\nTIME SERIES COMPLETENESS")

city_years = (df.groupby("city")["year"].nunique().sort_values())

print(city_years.describe())

print("\nCities with incomplete records:")

print(city_years[city_years < df["year"].nunique()])

# DATA QUALITY SUMMARY

print("\nDATA QUALITY SUMMARY")

print(f"Rows                 : {df.shape[0]:,}")
print(f"Columns              : {df.shape[1]}")
print(f"Missing Values       : {df.isnull().sum().sum()}")
print(f"Duplicate Rows       : {df.duplicated().sum()}")
print(f"Cities               : {df['city'].nunique()}")
print(f"Countries            : {df['iso_alpha3'].nunique()}")
print(f"Years                : {df['year'].min()} - {df['year'].max()}")

# DATA CLEANING
# All cleaning operations will be performed on a copy to preserve the raw dataset

df_clean = df.copy()

# MISSING VALUES
# for Numerical Variables - Missing values are replaced with the median.
# Why Median? Climate variables contains skewed distributions. The median is more robust to extreme values than the mean.
# for Categorical Variables
# Missing values are replaced with the mode

# Numerical Columns
numeric_cols = df_clean.select_dtypes(include=np.number).columns

for col in numeric_cols:

    if df_clean[col].isnull().sum() > 0:

        median_value = df_clean[col].median()

        df_clean[col].fillna(median_value, inplace=True)

# Categorical Columns
categorical_cols = df_clean.select_dtypes(include="object").columns

for col in categorical_cols:

    if df_clean[col].isnull().sum() > 0:

        mode_value = df_clean[col].mode()[0]

        df_clean[col].fillna(mode_value, inplace=True)
        
print(f"Missing Values: {df_clean.isnull().sum().sum()}")

# SORT DATASET
# Time-series feature engineering requires observations to be ordered chronologically within each city.
#This step is essential before creating: Lag Features, Rolling Statistics, Future Targets
df_clean = df_clean.sort_values(by=["city", "year"]).reset_index(drop=True)

# PREVENT DATA LEAKAGE & CREATE FORECAST TARGETS
# CREATE A COPY FOR MODEL DATASET
df_model = df_clean.copy()

leakage_features = ["temp_yoy_change", "temp_5yr_mean"]
df_model.drop(columns=leakage_features, inplace=True)
print("\nRemaining Features:", df_model.shape[1])

# CREATE FORECAST TARGETS

# Predict next year's rainfall
df_model["target_rainfall"] = (df_model.groupby("city")["precip_total_mm"].shift(-1))

plt.figure(figsize=(8,4))
df_model["target_rainfall"].hist(bins=40)
plt.title("Distribution of Target Rainfall")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.show()

# REMOVE ROWS WITHOUT FUTURE TARGETS
rows_before = len(df_model)

df_model.dropna(subset=["target_rainfall"], inplace=True)

rows_after = len(df_model)

print("\nRainfall Target Summary:")
print(df_model["target_rainfall"].describe())

print("\nTARGET CLEANING")

print(f"Rows Before : {rows_before:,}")
print(f"Rows After  : {rows_after:,}")
print(f"Rows Removed: {rows_before - rows_after:,}")

# VERIFY FORECAST TARGETS
print(df_model[["city", "year", "precip_total_mm", "target_rainfall"]].head(10))

# VERIFY IF LEAKAGE FEATURES HAS BEEN REMOVED
print("\nVERIFY LEAKAGE FEATURES")
removed_features = ["temp_yoy_change","temp_5yr_mean"]

for feature in removed_features:

    if feature in df_model.columns:
        print(f"{feature} : Still Exists")
    else:
        print(f"{feature} : Successfully Removed")
        
# FEATURE ENGINEERING DATASET

df_features = df_model.copy()

# LAG FEATURES - Previous years influence future rainfall.

lag_variables = ["precip_total_mm", "rh_mean_pct", "pressure_mean_kpa", "solar_total_mj", "wind_mean_ms"]

for variable in lag_variables:

    for lag in [1,2,3,4,5]:

        df_features[f"{variable}_lag{lag}"] = (df_features.groupby("city")[variable].shift(lag))

# ROLLING STATISTICS - Use ONLY previous observations to prevent leakage.

for window in [3,5,7]:

    # Rainfall Mean
    df_features[f"rain_roll{window}_mean"] = (df_features.groupby("city")["precip_total_mm"]
                                              .transform(lambda x: x.shift(1).rolling(window).mean()))

    # Rainfall Standard Deviation
    df_features[f"rain_roll{window}_std"] = (df_features.groupby("city")["precip_total_mm"]
                                             .transform(lambda x: x.shift(1).rolling(window).std()))

    # Humidity Mean
    df_features[f"humidity_roll{window}"] = (df_features.groupby("city")["rh_mean_pct"]
                                             .transform(lambda x: x.shift(1).rolling(window).mean()))

    # Pressure Mean
    df_features[f"pressure_roll{window}"] = (df_features.groupby("city")["pressure_mean_kpa"]
                                             .transform(lambda x: x.shift(1).rolling(window).mean()))

    # Solar Mean
    df_features[f"solar_roll{window}"] = (df_features.groupby("city")["solar_total_mj"]
                                          .transform(lambda x: x.shift(1).rolling(window).mean()))

    # Wind Mean
    df_features[f"wind_roll{window}"] = (df_features.groupby("city")["wind_mean_ms"]
                                         .transform(lambda x: x.shift(1).rolling(window).mean()))

# TREND FEATURES

df_features["rainfall_trend"] = (df_features["precip_total_mm_lag1"] - df_features["precip_total_mm_lag2"])

df_features["rainfall_acceleration"] = (df_features["precip_total_mm_lag1"] - 2 * df_features["precip_total_mm_lag2"] 
                                        + df_features["precip_total_mm_lag3"])

df_features["humidity_trend"] = (df_features["rh_mean_pct_lag1"] - df_features["rh_mean_pct_lag2"])

df_features["pressure_trend"] = (df_features["pressure_mean_kpa_lag1"] - df_features["pressure_mean_kpa_lag2"])

df_features["solar_trend"] = (df_features["solar_total_mj_lag1"] - df_features["solar_total_mj_lag2"])

df_features["wind_trend"] = (df_features["wind_mean_ms_lag1"] - df_features["wind_mean_ms_lag2"])

# TIME FEATURES

df_features["years_since_1990"] = (df_features["year"] - 1990)

# GEOGRAPHIC FEATURES

geographic_features = ["latitude", "longitude", "continent", "iso_alpha3"]

# REMOVE ROWS CREATED BY LAGS

rows_before = len(df_features)
df_features.dropna(inplace=True)
rows_after = len(df_features)

print("\nENGINEERED FEATURES")
print(f"Rows    : {rows_after:,}")
print(f"Columns : {df_features.shape[1]}")
print()
print(df_features.describe().T.head())

print()
print(f"Rows Before : {rows_before:,}")
print(f"Rows After  : {rows_after:,}")
print(f"Rows Removed: {rows_before-rows_after:,}")

# VERIFY ENGINEERED FEATURES

new_features = [
    col
    for col in df_features.columns   
    if ("lag" in col or "roll" in col or "trend" in col or "acceleration" in col or "years_since_1990" in col)
]

print()
print(f"Number of engineered features : {len(new_features)}")
print()
print(df_features[new_features].head())

# VISUALIZE TARGET
plt.figure(figsize=(10,5))
plt.plot(df_features.groupby("year")["precip_total_mm"].mean())
plt.title("Average Annual Rainfall")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.show()

# FEATURE SELECTION DATASET
df_final = df_features.copy()

# TARGET VARIABLE
y_rain = df_final["target_rainfall"]

#RAINFALL PREDICTOR MATRIX - Remove identifiers and target columns only.

rainfall_drop = ["target_rainfall", "city", "country", "iso_alpha3", "year"]

X_rain = df_final.drop(columns=rainfall_drop, errors="ignore")

# TIME SERIES CROSS VALIDATION
tscv = TimeSeriesSplit(n_splits=5)

# CROSS-VALIDATED FEATURE IMPORTANCE
feature_scores = pd.Series(0, index=X_rain.columns, dtype=float)

for train_index, valid_index in tscv.split(X_rain):

    X_train_cv = X_rain.iloc[train_index].copy()

    y_train_cv = y_rain.iloc[train_index]

    # Encoded categorical variables
    categorical_cols = X_train_cv.select_dtypes(include=["object", "category"]).columns
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_train_cv[categorical_cols] = encoder.fit_transform(X_train_cv[categorical_cols])

    # Trained Random Forest
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train_cv, y_train_cv)
    feature_scores += rf.feature_importances_

feature_scores /= tscv.get_n_splits()

# SORTING FEATURES
feature_scores = feature_scores.sort_values(ascending=False)

print()
print("AVERAGE FEATURE IMPORTANCE")
print(feature_scores.head(40))

# AUTOMATIC FEATURE SELECTION
selected_features = feature_scores.head(30).index.tolist()
X_rain = X_rain[selected_features]

# FEATURE IMPORTANCE PLOT
plt.figure(figsize=(10,10))
feature_scores.head(30).sort_values().plot.barh()
plt.title("Cross-Validated Rainfall Feature Importance")
plt.show()

print()
print("\nNumber of selected rainfall features")
print(len(selected_features))
print()
print(selected_features)

# VERIFY RAINFALL FEATURES
print()
print("\nRainfall Features Selected")
for i, feature in enumerate(selected_features, start=1):
    
    print(f"{i:02d}. {feature}")

# VERIFY DATASET
print()
print("\nDATASET SHAPE")
print(f"Predictors : {X_rain.shape}")
print(f"Target     : {y_rain.shape}")

print()
print("\nMISSING VALUES")
print(X_rain.isnull().sum().sum())

print()
print("\nFINAL DATASET PREVIEW")
print(X_rain.head())


# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
TRAIN_END = 2017
VALID_END = 2020

train_mask = df_final["year"] <= TRAIN_END
validation_mask = ((df_final["year"] > TRAIN_END) & (df_final["year"] <= VALID_END))
test_mask = (df_final["year"] > VALID_END)

# RAINFALL DATASETS
X_train_rain = X_rain.loc[train_mask].copy()
X_valid_rain = X_rain.loc[validation_mask].copy()
X_test_rain = X_rain.loc[test_mask].copy()

y_train_rain = y_rain.loc[train_mask].copy()
y_valid_rain = y_rain.loc[validation_mask].copy()
y_test_rain = y_rain.loc[test_mask].copy()


print("\nDATA SPLIT")
print(f"Training Samples   : {len(X_train_rain):,}")
print(f"Validation Samples : {len(X_valid_rain):,}")
print(f"Testing Samples    : {len(X_test_rain):,}")

# ORDINAL ENCODING
def encode_features(train_df, valid_df, test_df):
    train = train_df.copy()
    valid = valid_df.copy()
    test = test_df.copy()
    
    categorical_columns = train.select_dtypes(include=["object", "category"]).columns
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    if len(categorical_columns) > 0:
        
        train[categorical_columns] = encoder.fit_transform(train[categorical_columns])
        valid[categorical_columns] = encoder.transform(valid[categorical_columns])
        test[categorical_columns] = encoder.transform(test[categorical_columns])
        
    return train, valid, test, encoder


(X_train_rain_encoded, X_valid_rain_encoded, X_test_rain_encoded, rain_encoder) = encode_features(X_train_rain, X_valid_rain, X_test_rain)

# FEATURE SCALING

def scale_features(train_df, valid_df, test_df):

    train = train_df.copy()
    valid = valid_df.copy()
    test = test_df.copy()

    numeric_columns = train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    train[numeric_columns] = scaler.fit_transform(train[numeric_columns])
    valid[numeric_columns] = scaler.transform(valid[numeric_columns])
    test[numeric_columns] = scaler.transform(test[numeric_columns])

    return train, valid, test, scaler


(X_train_rain_scaled, X_valid_rain_scaled, X_test_rain_scaled, rain_scaler) = scale_features( X_train_rain_encoded, X_valid_rain_encoded, X_test_rain_encoded)

# SAVED PREPROCESSING OBJECTS
joblib.dump(rain_encoder, "rainfall_encoder.pkl")
joblib.dump(rain_scaler, "rainfall_scaler.pkl")

print("\nPREPROCESSING COMPLETE")
print("Rainfall encoder saved.")
print("Rainfall scaler saved.")

# PERSISTENCE BASELINE
print("\nPERSISTENCE BASELINE")

# Predict next year's rainfall using this year's rainfall
baseline_predictions = X_test_rain["precip_total_mm"]
baseline_mae = mean_absolute_error(y_test_rain, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error( y_test_rain, baseline_predictions))
baseline_r2 = r2_score(y_test_rain, baseline_predictions)
baseline_mape = mean_absolute_percentage_error(y_test_rain, baseline_predictions)

print(f"Baseline MAE   : {baseline_mae:.4f}")
print(f"Baseline RMSE  : {baseline_rmse:.4f}")
print(f"Baseline R²    : {baseline_r2:.4f}")
print(f"Baseline MAPE  : {baseline_mape:.4f}")


# MODEL TRAINING
print("\nMODEL TRAINING - RAINFALL")

# BASELINE MODELS
linear_rain = LinearRegression()
tree_rain = DecisionTreeRegressor(max_depth=10, random_state=42)

# RANDOM FOREST SEARCH
rf_parameters = {"n_estimators": randint(500, 1500), "max_depth": [8,10,12,15,20,None], "min_samples_split": randint(2,8),
                 "min_samples_leaf": randint(1,5), "max_features":[ "sqrt", 0.5, 0.7]}

# XGBOOST SEARCH
xgb_parameters = {"n_estimators": randint(300,1200), "learning_rate": loguniform(0.005,0.15), "max_depth": randint(3,10), 
                  "min_child_weight": randint(1,10), "gamma": uniform(0,2), "subsample": uniform(0.6,0.4), 
                  "colsample_bytree": uniform(0.6,0.4), "reg_alpha": loguniform(1e-4,1), "reg_lambda": loguniform(0.5,10),
                  "tree_method":["hist"], "max_bin":[256]}

# RANDOM FOREST TUNING
print("\nTUNING RANDOM FOREST")

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(estimator=rf_base, param_distributions=rf_parameters, n_iter=40, scoring="r2", cv=tscv,
                               random_state=42, n_jobs=-1, verbose=2)
rf_search.fit(X_train_rain_encoded, y_train_rain)

rf_rain = rf_search.best_estimator_

print("\nBEST RF PARAMETERS")
print(rf_search.best_params_)
print("\nBEST RF CV SCORE")
print(rf_search.best_score_)

# XGBOOST TUNING
print("\nTUNING XGBOOST")

xgb_base = XGBRegressor( objective="reg:squarederror", random_state=42, n_jobs=-1)
xgb_search = RandomizedSearchCV(estimator=xgb_base, param_distributions=xgb_parameters, n_iter=40, scoring="r2",
                                cv=tscv, random_state=42, n_jobs=-1, verbose=2)
xgb_search.fit(X_train_rain_encoded, y_train_rain)

best_xgb = xgb_search.best_estimator_

print("\nBEST XGB PARAMETERS")
print(xgb_search.best_params_)
print("\nBEST XGB CV SCORE")
print(xgb_search.best_score_)

# EARLY STOPPING
best_xgb.fit(X_train_rain_encoded, y_train_rain, eval_set=[(X_valid_rain_encoded, y_valid_rain)], verbose=False)

xgb_rain = best_xgb

# TRAINING SIMPLE MODEL
linear_rain.fit(X_train_rain_scaled, y_train_rain)
tree_rain.fit(X_train_rain_encoded, y_train_rain)

# RF already trained by RandomizedSearch, XGB already trained above

# SAVE BEST MODELS
joblib.dump(rf_rain, "best_random_forest_rainfall.pkl")
joblib.dump(xgb_rain, "best_xgboost_rainfall.pkl")

# MODEL COLLECTION
rainfall_models = {"Linear Regression": linear_rain, "Decision Tree": tree_rain, "Random Forest": rf_rain, "XGBoost": xgb_rain}

print()
print("Rainfall models trained successfully.")

# MODEL EVALUATION

def evaluate_model(model, X, y_true):

    """
    Evaluate a regression model.
    """

    predictions = model.predict(X)

    mae = mean_absolute_error(y_true, predictions)

    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    r2 = r2_score(y_true, predictions)

    mape = mean_absolute_percentage_error(y_true, predictions)

    return predictions, mae, rmse, r2, mape

# VALIDATION RESULTS

print("\nVALIDATION RESULTS")
validation_results = []
validation_models = {"Linear Regression": linear_rain, "Decision Tree": tree_rain, "Random Forest": rf_rain, "XGBoost": xgb_rain}

for name, model in validation_models.items():

    if name == "Linear Regression":
        X_val = X_valid_rain_scaled

    else:
        X_val = X_valid_rain_encoded

    predictions, mae, rmse, r2, mape = evaluate_model(model, X_val, y_valid_rain)

    validation_results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape})

validation_results = pd.DataFrame(validation_results)
validation_results = validation_results.sort_values(by="RMSE")

print(validation_results)

# BEST VALIDATION MODEL

best_model_name = validation_results.iloc[0]["Model"]

print()
print("\nBEST VALIDATION MODEL")
print(best_model_name)

rainfall_models = {"Linear Regression": linear_rain, "Decision Tree": tree_rain, "Random Forest": rf_rain, "XGBoost": xgb_rain}

best_rainfall_model = rainfall_models[best_model_name]

# FINAL TEST EVALUATION

print("\nFINAL TEST PERFORMANCE")

if best_model_name == "Linear Regression":
    X_test_final = X_test_rain_scaled

else:
    X_test_final = X_test_rain_encoded

predictions, mae, rmse, r2, mape = evaluate_model(best_rainfall_model, X_test_final, y_test_rain)

print()
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.4f}")

# MODEL VS BASELINE

print("\nMODEL VS PERSISTENCE BASELINE")

comparison = pd.DataFrame({"Model":["Persistence Baseline", "Best ML Model"], "RMSE":[baseline_rmse, rmse], "R²":[baseline_r2, r2]})

print(comparison)

# ACTUAL VS PREDICTED
plt.figure(figsize=(8,8))
plt.scatter(y_test_rain, predictions, alpha=0.7)

plt.plot([y_test_rain.min(), y_test_rain.max()], [y_test_rain.min(), y_test_rain.max()], "r--")
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs Predicted Rainfall")
plt.show()

# RESIDUAL PLOT
residuals = y_test_rain - predictions

plt.figure(figsize=(8,6))
plt.scatter(predictions, residuals, alpha=0.7)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Rainfall")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

# RESIDUAL DISTRIBUTION
plt.figure(figsize=(8,6))
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# PERMUTATION IMPORTANCE

if best_model_name != "Linear Regression":

    result = permutation_importance(best_rainfall_model, X_test_final, y_test_rain, scoring="r2", random_state=42, n_repeats=20)

    importance = pd.DataFrame({"Feature": X_test_final.columns, "Importance": result.importances_mean})

    importance = importance.sort_values("Importance", ascending=False)

    print()
    print("\nTOP 20 IMPORTANT FEATURES")
    print(importance.head(20))
    plt.figure(figsize=(10,8))
    plt.barh(importance["Feature"][:20], importance["Importance"][:20])
    plt.gca().invert_yaxis()
    plt.title("Permutation Importance")
    plt.show()


# SAVED FINAL ARTIFACTS

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

joblib.dump(best_rainfall_model, MODEL_DIR / "rainfall_forecasting_model.pkl")

joblib.dump(rain_encoder, MODEL_DIR / "rainfall_encoder.pkl")

joblib.dump(rain_scaler, MODEL_DIR / "rainfall_scaler.pkl")

joblib.dump(selected_features, MODEL_DIR / "rainfall_selected_features.pkl")

joblib.dump(feature_scores, MODEL_DIR / "rainfall_feature_importance.pkl")

df_final.to_csv(DATA_DIR / "rainfall_training_dataset.csv", index=False)

print()
print("Rainfall forecasting completed successfully.")
print(f"Models saved to: {MODEL_DIR.resolve()}")
print(f"Dataset saved to: {DATA_DIR.resolve()}")
