import joblib
import pandas as pd
import numpy as np

from pathlib import Path


class ForecastEngine:
    """
    Rainfall forecasting engine.

    Responsibilities:
        • Load trained model
        • Load preprocessing objects
        • Load historical dataset
        • Prepare features
        • Forecast future rainfall
    """

    def __init__(self):

        # Project directories
        self.BASE_DIR = Path(__file__).resolve().parent
        self.MODEL_DIR = self.BASE_DIR / "models"
        self.DATA_DIR = self.BASE_DIR / "data"

        # Load trained model
        self.model = joblib.load(self.MODEL_DIR / "rainfall_forecasting_model.pkl")

        # Load encoder
        self.encoder = joblib.load(self.MODEL_DIR / "rainfall_encoder.pkl")

        # Load selected features
        self.selected_features = joblib.load(self.MODEL_DIR /"rainfall_selected_features.pkl")

        # Load historical dataset
        self.df = pd.read_csv(self.DATA_DIR / "rainfall_training_dataset.csv")

        print("Rainfall Forecast Engine initialized successfully.")
        
    # Available Cities
    def get_available_cities(self):
        """
        Return a sorted list of all cities
        available in the dataset.
        """
        cities = sorted(self.df["city"].unique())
        
        return cities    
    
    # City History
    def get_city_history(self, city):
        """
        Return the historical rainfall data
        for one city.
        """
        city_df = self.df[self.df["city"] == city].copy()
        city_df = city_df.sort_values("year")
        city_df.reset_index(drop=True, inplace=True)
        
        return city_df
    
    # Prepare Features
    def prepare_features(self, row):
        """
        Prepare model input and validate
        required features.
        """
        X = pd.DataFrame([row])

        missing = [
            feature
            for feature in self.selected_features
            if feature not in X.columns
        ]

        if missing:
            raise ValueError(f"Missing required features:\n{missing}")
        
        X = X[self.selected_features]

        return X
    
    # Encode Features
    def encode_data(self, X):
        """
        Encode categorical variables.
        """
        X = X.copy()
        categorical_columns = (X.select_dtypes(include=["object", "category"]).columns)

        if len(categorical_columns) > 0:
            X[categorical_columns] = (self.encoder.transform(X[categorical_columns]))

        return X
    
    # Feature Engineering

    def engineer_future_features(self, city_df):
        """
        Recalculate engineered features after adding
        a synthetic future observation.
        """
        city_df = city_df.copy()
        
        # Rainfall Lag Features
        for lag in [1, 2, 3, 4, 5]:
            city_df[f"precip_total_mm_lag{lag}"] = (city_df["precip_total_mm"].shift(lag))

        # Other Climate Variable Lags
        climate_variables = ["rh_mean_pct", "pressure_mean_kpa", "solar_total_mj", "wind_mean_ms"]

        for variable in climate_variables:
            
            for lag in [1, 2, 3, 4, 5]:
                city_df[f"{variable}_lag{lag}"] = (city_df[variable].shift(lag))
                
        # Rolling Statistics

        for window in [3, 5, 7]:
            
            #Rainfall
            city_df[f"rain_roll{window}_mean"] = (city_df["precip_total_mm"].shift(1).rolling(window).mean())
            city_df[f"rain_roll{window}_std"] = (city_df["precip_total_mm"].shift(1).rolling(window).std())

            # Humidity
            city_df[f"humidity_roll{window}"] = (city_df["rh_mean_pct"] .shift(1) .rolling(window) .mean())

            # Pressure
            city_df[f"pressure_roll{window}"] = ( city_df["pressure_mean_kpa"].shift(1).rolling(window).mean())

            # Solar
            city_df[f"solar_roll{window}"] = (city_df["solar_total_mj"].shift(1).rolling(window).mean())

            # Wind
            city_df[f"wind_roll{window}"] = (city_df["wind_mean_ms"].shift(1).rolling(window).mean())
            
        # Trend Features
        city_df["rainfall_trend"] = (city_df["precip_total_mm_lag1"] - city_df["precip_total_mm_lag2"])
        
        city_df["rainfall_acceleration"] = (city_df["precip_total_mm_lag1"] - 2 * city_df["precip_total_mm_lag2"] 
                                            + city_df["precip_total_mm_lag3"])
        
        city_df["humidity_trend"] = (city_df["rh_mean_pct_lag1"] - city_df["rh_mean_pct_lag2"])

        city_df["pressure_trend"] = (city_df["pressure_mean_kpa_lag1"] - city_df["pressure_mean_kpa_lag2"])

        city_df["solar_trend"] = (city_df["solar_total_mj_lag1"] - city_df["solar_total_mj_lag2"])

        city_df["wind_trend"] = (city_df["wind_mean_ms_lag1"] - city_df["wind_mean_ms_lag2"])

        # Time Feature
        city_df["years_since_1990"] = (city_df["year"] - 1990)

        return city_df
    
     # Create Future Observation

    def create_future_row(self, city_df, prediction):
        """
        Create one synthetic future observation
        using the predicted rainfall.
        """
        latest = city_df.iloc[-1].copy()

        future = latest.copy()
        future["year"] += 1

        # Update rainfall prediction
        future["precip_total_mm"] = prediction

        # Carry forward climate variables
        climate_variables = ["rh_mean_pct", "pressure_mean_kpa", "solar_total_mj", "wind_mean_ms"]

        for variable in climate_variables:
            future[variable] = latest[variable]

        # Append
        city_df = pd.concat([city_df, pd.DataFrame([future])],ignore_index=True)

        # Recalculate engineered features
        city_df = self.engineer_future_features(city_df)

        return city_df
    
    
    # Predict Next Year
    def predict_next_year(self, city):
        validation = self.validate_features(city)

        if not validation["is_valid"]:
            raise ValueError(
                "Feature validation failed.\n\n"
                f"Missing:\n"
                f"{validation['missing_features']}"
            )

        city_df = self.get_city_history(city)
        latest = city_df.iloc[-1]

        X = self.prepare_features(latest)
        X = self.encode_data(X)

        prediction = self.model.predict(X)[0]

        return prediction
    
    # Recursive Forecast
    def recursive_forecast(self, city, end_year=2027):
        """
        Forecast rainfall recursively from the
        latest historical year up to end_year.
        """
        city_df = self.get_city_history(city)

        forecasts = []

        while city_df.iloc[-1]["year"] < end_year:

            latest = city_df.iloc[-1]

            X = self.prepare_features(latest)
            X = self.encode_data(X)

            prediction = self.model.predict(X)[0]
            next_year = int(latest["year"] + 1)
            forecasts.append({"City": city, "Year": next_year, "Predicted Rainfall (mm)": round(float(prediction), 2)})
            city_df = self.create_future_row(city_df, prediction)

        return pd.DataFrame(forecasts)
    
    # Forecast Between Years
    def forecast_between_years(self, city, start_year, end_year):
        """
        Return forecasts between two years.
        """
        forecasts = self.recursive_forecast(city, end_year)
        forecasts = forecasts[forecasts["Year"] >= start_year]
        forecasts.reset_index(drop=True, inplace=True)

        return forecasts
    
    # Validate Feature Consistency
    def validate_features(self, city):
        """
        Ensure that all selected model features
        exist after feature engineering.
        """
        city_df = self.get_city_history(city)

        # Recompute engineered features
        city_df = self.engineer_future_features(city_df)
        available_features = set(city_df.columns)
        required_features = set(self.selected_features)

        missing_features = sorted(required_features - available_features)
        extra_features = sorted(available_features - required_features)

        return {
            "is_valid": len(missing_features) == 0,
            "missing_features": missing_features,
            "extra_features": extra_features
        }
    

if __name__ == "__main__":

    engine = ForecastEngine()

    city = engine.get_available_cities()[0]

    print("PROFESSIONAL RAINFALL FORECAST ENGINE")
    
    # Validate Features
    validation = engine.validate_features(city)

    print("\nFeature Validation")

    if validation["is_valid"]:
        print("PASS")

    else:
        print("FAIL")
        print(validation["missing_features"])

    # Predict One Year
    prediction = engine.predict_next_year(city)

    print(f"\n2025 Prediction ({city})")
    print(f"{prediction:.2f} mm")

    # Recursive Forecast
    forecasts = engine.forecast_between_years(city, 2025, 2030)

    print("\nForecasts")
    print(forecasts)
    print("\nForecast engine completed successfully.")