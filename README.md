# RBI-Consumer-Price-Index-forecasting-using-ML
This Project is a Test project which explain how CPI affects economic policies from microeconomic(Salary,cost of essential commodities ) and macroeconomic(Government bonds,forex,rupee valuation etc)
!pip install streamlit
# -*- coding: utf-8 -*-
"""
RBI CPI Inflation Forecasting Project - Streamlit Application

This application provides an interactive interface for forecasting the Consumer Price Index (CPI)
using the Prophet library and visualizing the results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# Set Streamlit page configuration
st.set_page_config(
    page_title="RBI CPI Inflation Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the CPI data."""
    try:
        df = pd.read_csv(file_path)
        # Convert 'Month' to datetime, coercing errors
        df['Month'] = pd.to_datetime(df['Month'], format='%b-%Y', errors='coerce')
        df.dropna(subset=['Month'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file '{file_path}' not found. Please ensure 'cpi_data_cleaned.csv' is in the same directory.")
        return pd.DataFrame()

def get_available_columns(df):
    """Extracts the list of forecastable CPI components."""
    # Columns are in the format: 'Category_Area_Metric'
    cpi_cols = [col for col in df.columns if 'Index' in col and col not in ['General_Index__Combined__Index']]

    # Extract unique categories and areas
    categories = sorted(list(set([col.split('_')[0] for col in cpi_cols])))
    areas = sorted(list(set([col.split('_')[1] for col in cpi_cols])))

    # Add the main General Index back
    categories.append('General')
    areas.append('Combined')

    return categories, areas

def run_prophet_forecast(df_input, periods):
    """Runs the Prophet model on the prepared data."""
    # Prophet requires columns 'ds' (datestamp) and 'y' (value)
    df_prophet = df_input[['ds', 'y']].copy()

    # Initialize and fit the model
    model = Prophet(
        yearly_seasonality=True,
        # Add a custom holiday list for India for better accuracy if available
        # holidays=indian_holidays_df,
        changepoint_prior_scale=0.05 # Default is 0.05, can be tuned
    )

    model.fit(df_prophet)

    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='M')

    # Make predictions
    forecast = model.predict(future)

    return model, forecast, df_prophet

def calculate_yoy_inflation(full_index_series):
    """Calculates the Year-on-Year inflation rate."""
    # Inflation Rate (t) = [(Index(t) / Index(t-12)) - 1] * 100
    yoy_inflation = (full_index_series / full_index_series.shift(12) - 1) * 100
    return yoy_inflation

# --- Main Application ---

st.title("🇮🇳 RBI CPI Inflation Forecasting Dashboard")
st.markdown("A time series forecasting application for the Consumer Price Index (CPI) using the **Prophet** model.")

# 1. Load Data
df_raw = load_data('cpi_data_cleaned.csv')

if df_raw.empty:
    st.stop()

# Get unique states and CPI components
states = sorted(df_raw['State_UT'].unique().tolist())
categories, areas = get_available_columns(df_raw)

# --- Sidebar for User Input ---
st.sidebar.header("Forecasting Parameters")

selected_state = st.sidebar.selectbox(
    "Select State/Union Territory:",
    options=states,
    index=states.index('ALL INDIA') if 'ALL INDIA' in states else 0
)

selected_category = st.sidebar.selectbox(
    "Select CPI Component:",
    options=categories,
    index=categories.index('General') if 'General' in categories else 0
)

selected_area = st.sidebar.selectbox(
    "Select Area Type:",
    options=['Combined', 'Rural', 'Urban'],
    index=0
)

forecast_periods = st.sidebar.slider(
    "Forecast Horizon (Months):",
    min_value=6,
    max_value=36,
    value=12,
    step=6
)

# Construct the column name based on user selection
if selected_category == 'General':
    target_col = f"General_Index__{selected_area}__Index"
else:
    target_col = f"{selected_category}_and_beverages__{selected_area}__Index" # Assuming 'and_beverages' for simplicity based on the data structure

if target_col not in df_raw.columns:
    st.sidebar.warning(f"Selected combination '{target_col}' not found. Defaulting to 'General_Index__Combined__Index'.")
    target_col = "General_Index__Combined__Index"
    selected_category = 'General'
    selected_area = 'Combined'

# --- Data Preparation for Forecasting ---

df_filtered = df_raw[df_raw['State_UT'] == selected_state].copy()
df_prophet_input = df_filtered[['Month', target_col]].copy()
df_prophet_input.columns = ['ds', 'y']

# Convert 'y' to numeric and drop NaNs
df_prophet_input['y'] = pd.to_numeric(df_prophet_input['y'], errors='coerce')
df_prophet_input.dropna(subset=['y'], inplace=True)

st.sidebar.info(f"Forecasting **{selected_category} CPI Index ({selected_area})** for **{selected_state}**.")

# --- Run Forecasting ---
if not df_prophet_input.empty:
    with st.spinner("Running Prophet Model and Generating Forecast..."):
        model, forecast, df_historical = run_prophet_forecast(df_prophet_input, forecast_periods)

    st.success("Forecasting Complete!")

    # --- Visualization ---

    st.header("1. CPI Index Forecast")

    # Plotly for interactive visualization
    fig = plot_plotly(model, forecast)
    fig.update_layout(
        title=f'{selected_category} CPI Index Forecast for {selected_state} ({selected_area})',
        xaxis_title="Date",
        yaxis_title="CPI Index (Base 2012=100)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Inflation Rate Calculation and Visualization ---

    # Combine historical and forecasted index values
    full_index = pd.concat([df_historical.set_index('ds')['y'], forecast.set_index('ds')['yhat']], axis=1)
    full_index.columns = ['Historical_Index', 'Forecasted_Index']
    full_index['Combined_Index'] = full_index['Historical_Index'].fillna(full_index['Forecasted_Index'])

    # Calculate Year-on-Year Inflation
    full_index['YoY_Inflation_Pct'] = calculate_yoy_inflation(full_index['Combined_Index'])

    # Filter for the forecasted period's inflation
    forecast_inflation = full_index[full_index.index > df_historical['ds'].max()].copy()

    st.header("2. Year-on-Year Inflation Forecast")

    # Plotly for interactive inflation visualization
    fig_inf = px.line(
        full_index.reset_index(),
        x='ds',
        y='YoY_Inflation_Pct',
        title=f'Year-on-Year Inflation Rate Forecast for {selected_state} ({selected_area})',
        labels={'ds': 'Date', 'YoY_Inflation_Pct': 'Inflation Rate (%)'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    # Highlight the forecasted period
    fig_inf.add_vline(x=df_historical['ds'].max(), line_width=2, line_dash="dash", line_color="red", annotation_text="Forecast Start")

    fig_inf.update_layout(hovermode="x unified")
    st.plotly_chart(fig_inf, use_container_width=True)

    # --- Model Components ---
    st.header("3. Model Components")
    st.markdown("Prophet decomposes the time series into trend and seasonality components.")

    # Use Prophet's built-in component plot
    fig_comp = model.plot_components(forecast)
    st.pyplot(fig_comp)

    # --- Raw Forecast Data ---
    st.header("4. Raw Forecast Data")
    st.markdown(f"Forecast for the next {forecast_periods} months:")

    # Prepare final table for display
    final_forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods).copy()
    final_forecast_table['YoY_Inflation_Pct'] = forecast_inflation['YoY_Inflation_Pct'].head(forecast_periods).values
    final_forecast_table.columns = ['Date', 'Forecasted Index (yhat)', 'Lower Bound', 'Upper Bound', 'YoY Inflation (%)']

    # Format the columns for better readability
    final_forecast_table['Date'] = final_forecast_table['Date'].dt.strftime('%Y-%m-%d')
    for col in ['Forecasted Index (yhat)', 'Lower Bound', 'Upper Bound', 'YoY Inflation (%)']:
        final_forecast_table[col] = final_forecast_table[col].round(2)

    st.dataframe(final_forecast_table, use_container_width=True)

else:
    st.warning("No valid data points found for the selected parameters to run the forecast.")

# --- Instructions for Running ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to Run This App:**
1. Save this code as `streamlit_app.py`.
2. Ensure `cpi_data_cleaned.csv` is in the same directory.
3. Run in your terminal: `streamlit run streamlit_app.py`
""")
