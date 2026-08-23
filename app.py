import streamlit as st
import pandas as pd
import plotly.express as px

from forecast import ForecastEngine

#Streamlit Configuration
st.set_page_config(page_title="Rainfall Forecasting System", page_icon="🌧️", 
                   layout="wide", initial_sidebar_state="expanded")

#Load Forecast Engine - caching it
@st.cache_resource
def load_engine():
    
    return ForecastEngine()

engine = load_engine()

#CSS

st.markdown("""
<style>

.main-header{
    font-size:42px;
    font-weight:bold;
    color:#0B5394;
}

.sub-header{
    font-size:24px;
    color:#444;
}

.metric-card{
    background-color:#F7F9FC;
    padding:20px;
    border-radius:12px;
    border:1px solid #DDD;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

#Home Page
st.markdown(
    '<p class="main-header">🌧️ Rainfall Forecasting System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-header">Machine Learning Forecast Dashboard</p>',
    unsafe_allow_html=True
)

#Project Overview
st.write("""
This application predicts future rainfall using a trained
Random Forest Machine Learning model.

Features:

- Predict rainfall up to 2030
- Recursive forecasting
- Interactive charts
- Download results as CSV
- Portfolio-quality dashboard

Dataset: NASA POWER Climate Risk Dataset

Model: Random Forest Regressor
""")

#Sidebar
st.sidebar.header("Forecast Settings")

#City Selector
cities = engine.get_available_cities()
selected_city = st.sidebar.selectbox("Select City", cities)

#Year Selector
forecast_year = st.sidebar.slider("Forecast Until", min_value=2025, max_value=2030, value=2027)

#Forecast Button
run_forecast = st.sidebar.button("Generate Forecast")

#Test
if run_forecast:

    with st.spinner("Generating rainfall forecast..."):

        forecasts = engine.forecast_between_years(city=selected_city, start_year=2025, end_year=forecast_year)

    st.success("Forecast generated successfully!")
    
    # Dashboard Metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Forecast Years", len(forecasts))

    col2.metric("Average Rainfall", f"{forecasts['Predicted Rainfall (mm)'].mean():.2f} mm")

    col3.metric("Maximum Rainfall", f"{forecasts['Predicted Rainfall (mm)'].max():.2f} mm")

    col4.metric("Best Model", "Random Forest")
    
    st.subheader("Forecast Results")
    st.dataframe(forecasts, use_container_width=True)
    
    
    # Rainfall Trend Chart

    st.subheader("📈 Rainfall Trend")

    fig = px.line(forecasts, x="Year", y="Predicted Rainfall (mm)", markers=True, title=f"Projected Rainfall Trend for {selected_city}")

    fig.update_layout(xaxis_title="Year", yaxis_title="Rainfall (mm)", hovermode="x unified", template="plotly_white", height=500)

    st.plotly_chart(fig, use_container_width=True)

    # To Download Forecast prediction as CSV file

    st.subheader("📥 Download Forecast")

    csv = forecasts.to_csv(index=False).encode("utf-8")

    st.download_button(label="Download Predictions (CSV)", data=csv, file_name=f"{selected_city}_rainfall_forecast.csv", mime="text/csv")

    # Model Information

    st.subheader("ℹ️ Model Information")

    col1, col2 = st.columns(2)

    with col1: st.info(
            """
    Forecast Model

    - Algorithm: Random Forest Regressor
    - Forecast Type: Recursive Multi-Year Forecasting
    - Dataset: NASA POWER Climate Dataset
    - Target Variable: Annual Rainfall (mm)
    """
        )

    with col2:
        st.info(
            f"""
    Training Information

    - Historical Data: 1990–2024
    - Forecast Range: 2025–2030
    - Selected Features:
    {len(engine.selected_features)}
    - Forecast Engine:
    ForecastEngine
    """
        )
        
    # To Display Model Performance
    st.subheader("📊 Model Performance")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("MAE", "12.54 mm")
    metric2.metric("RMSE", "18.33 mm")
    metric3.metric("R² Score", "0.93")