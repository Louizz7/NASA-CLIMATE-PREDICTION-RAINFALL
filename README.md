 NASA Rainfall Forecasting System

An end-to-end machine learning project for forecasting annual rainfall using climate data from the NASA POWER Climate Risk Indices dataset.

The project covers the complete workflow:

Data preparation → Feature engineering → Time-series train/validation/test split → Model training → Model evaluation → Model selection → Recursive forecasting → Streamlit deployment


Project Overview

Rainfall variability creates planning challenges for agriculture, water-resource management, infrastructure, and climate-risk assessment.

This project develops a machine-learning forecasting system that uses historical climate observations to predict future annual rainfall.

The trained forecasting system can generate recursive forecasts from 2025 through 2030 for supported cities.

 Key capabilities

- Forecast annual rainfall for individual cities
- Generate recursive multi-year forecasts
- Recalculate lag, rolling, and trend features during recursive forecasting
- Use a trained Random Forest model
- Interactive Streamlit dashboard
- Interactive Plotly rainfall trend chart
- Download forecast results as CSV
- Display model information and evaluation metrics


 Dataset

Source: NASA POWER Climate Risk Indices

Historical period: 1990–2024

The project uses climate variables and engineered historical features to predict annual rainfall.

The processed dataset used by the forecasting application is:

```text
data/rainfall_training_dataset.csv
```

> The application uses the processed training dataset rather than requiring the original raw dataset at runtime.


Machine Learning Approach

Several regression models were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

Model performance was evaluated using:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² — Coefficient of Determination
- MAPE — Mean Absolute Percentage Error

The final model is selected from the validation results based on the lowest RMSE.

 Selected production model

Random Forest Regressor

The production artifact is stored as:

```text
models/rainfall_forecasting_model.pkl
```

Recursive Forecasting

The application uses recursive forecasting for future years.

The process is:

```text
2024 historical data
       ↓
Predict 2025
       ↓
Add predicted 2025 rainfall
       ↓
Recalculate lag/rolling/trend features
       ↓
Predict 2026
       ↓
Add predicted 2026 rainfall
       ↓
Recalculate dependent features
       ↓
Predict 2027
       ↓
...
       ↓
Predict 2030
```

This approach ensures that future predictions are generated using the model's own previous predictions rather than incorrectly treating future rainfall as known historical data.


Feature Engineering

The forecasting engine reconstructs dependent features for every recursive prediction step, including:

- Rainfall lag features
- Climate-variable lag features
- Rolling rainfall statistics
- Rolling humidity statistics
- Rolling pressure statistics
- Rolling solar statistics
- Rolling wind statistics
- Rainfall trend
- Rainfall acceleration
- Humidity trend
- Pressure trend
- Solar trend
- Wind trend
- Years since 1990

This is particularly important for multi-year forecasting because the engineered features for 2026 depend partly on the predicted 2025 observation.


Streamlit Application

The Streamlit application provides:

 Home page
Project overview and forecasting system description.

 Forecast controls
- City selector
- Forecast-year selector from 2025–2030
- One-click forecast generation

 Forecast results
- Forecast table
- Average predicted rainfall
- Maximum predicted rainfall
- Number of forecast years
- Best-model indicator

 Interactive visualization
Plotly rainfall trend chart with:

- Hover tooltips
- Zoom
- Pan
- Interactive controls

 Download
Forecast predictions can be downloaded as a CSV file.


Project Structure

NASA-CLIMATE-PREDICTION-RAINFALL/
│
├── app.py
├── forecast.py
├── NASA CLIMATE PREDICTION - RAINFALL.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── rainfall_training_dataset.csv
│
└── models/
    ├── rainfall_forecasting_model.pkl
    ├── rainfall_encoder.pkl
    ├── rainfall_scaler.pkl
    ├── rainfall_selected_features.pkl
    └── rainfall_feature_importance.pkl

If you want the original NASA CSV included in GitHub, it can be placed in `data/`; however, it is not required by the deployed application if `rainfall_training_dataset.csv` is already included.


Installation

 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd NASA-CLIMATE-PREDICTION-RAINFALL
```

 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the Streamlit Application

Do not run the application with:

```powershell
python app.py
```

Run it with:

```powershell
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```


Run the Training Script

The training script is:

```text
NASA CLIMATE PREDICTION - RAINFALL.py
```

Running the training script:

1. Loads the climate dataset
2. Cleans the data
3. Engineers forecasting features
4. Performs chronological splitting
5. Encodes categorical variables
6. Trains candidate models
7. Tunes selected models
8. Evaluates validation performance
9. Selects the best rainfall model
10. Performs final test evaluation
11. Calculates permutation importance
12. Saves the production artifacts

The saved artifacts are written to:

```text
models/
```

and the processed dataset is written to:

```text
data/rainfall_training_dataset.csv
```

Production Artifacts

The Streamlit application depends on the following artifacts:

| Artifact | Purpose |
|---|---|
| `rainfall_forecasting_model.pkl` | Final trained rainfall model |
| `rainfall_encoder.pkl` | Categorical feature encoder |
| `rainfall_scaler.pkl` | Saved numerical scaler |
| `rainfall_selected_features.pkl` | Exact model feature list |
| `rainfall_feature_importance.pkl` | Feature-importance information |
| `rainfall_training_dataset.csv` | Historical data used by the forecasting engine |

Keep these files synchronized with the version of the training script that created them.


Important Forecasting Consideration

Recursive forecasting becomes increasingly dependent on previous predictions as the forecast horizon increases.

For example:

```text
2025 prediction → influences 2026
2026 prediction → influences 2027
2027 prediction → influences 2028
```

Therefore, uncertainty generally increases as the forecast horizon becomes longer.

The 2025 forecast is expected to be more directly informed by observed historical data than the 2030 forecast.


Evaluation

The training workflow evaluates the final model using:

```text
MAE
RMSE
R²
MAPE
```

The exact final metrics should be taken from the output of the final training run rather than manually hard-coded into the application.


Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SciPy
- Joblib
- Matplotlib
- Seaborn
- Plotly
- Streamlit


Deployment

The application is designed to be deployable as a Streamlit application.

Before deployment, verify that:

- `app.py` imports `ForecastEngine` successfully
- `forecast.py` is valid and tested
- all required model artifacts exist
- `data/rainfall_training_dataset.csv` exists
- `requirements.txt` installs successfully
- the app runs with `python -m streamlit run app.py`
  

 👤 Author

Louis Mbagwu

Data Science / Machine Learning Project

Project Status

Status: Production-style portfolio project

Forecast horizon: 2025–2030

Production model: Random Forest Regressor

Deployment: Streamlit
