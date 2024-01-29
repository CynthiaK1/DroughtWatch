# pip install streamlit prophet pandas numpy scikit-learn plotly
import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import plotly.graph_objs as go

st.title('Forecasting with DroughtWatch')

# Load data
df_test = pd.read_csv('/Users/cynthia/Downloads/ProjectApp/imageclassificaltion/TimeSeriesVariableData.csv')
df_train = pd.read_csv("/Users/cynthia/Downloads/ProjectApp/imageclassificaltion/TimeSeriesVariableData.csv")

# Rename columns for Prophet
df_test = df_test.rename(columns={'forecast_value': 'y', 'DATE': 'ds'})
df_train = df_train.rename(columns={'forecast_value': 'y', 'DATE': 'ds'})
df_train['y_orig'] = df_train['y']
df_train['y'] = np.log(df_train['y'])
df = df_train.copy()

# Instantiate and fit Prophet model
model_new = Prophet()
model_new.add_regressor('Temparature')
model_new.add_regressor('Humidity')
model_new.fit(df_train)

# Define forecasting function
def forecast_and_plot(n_years):
    future_data = pd.DataFrame({'ds': pd.date_range(start=df_train['ds'].min(), periods=n_years * 365, freq='D')})
    future_data['ds'] = pd.to_datetime(future_data['ds'])

    # Add regressor columns for future data
    future_data['Temparature'] = 0  # You may need to replace 0 with appropriate values
    future_data['Humidity'] = 0  # You may need to replace 0 with appropriate values

    # Predict future values
    forecast_data = model_new.predict(future_data)

# Check if any predicted values are above 6
    if forecast_data['yhat'].max() > 6:
        st.warning('Early Warning: Prepare for drought!')
    else:
        st.success('Good news! No Drought soon')

# Transform back to original scale
    forecast_data_orig = forecast_data.copy()
    forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])


    # Plot components
    st.subheader('Forecast Components')
    st.write(model_new.plot_components(forecast_data_orig))

    # Plot forecast values
    st.subheader('Forecast Plot')
    fig = go.Figure()

    # Plot actual values
    actual_chart = go.Scatter(x=df_train['ds'], y=df_train['y_orig'], name='Actual', mode='lines+markers')
    fig.add_trace(actual_chart)

    # Plot predicted values
    predict_chart = go.Scatter(x=forecast_data['ds'], y=forecast_data_orig['yhat'], name='Predicted', mode='lines+markers')
    fig.add_trace(predict_chart)

    # Plot upper and lower bounds
    predict_chart_upper = go.Scatter(x=forecast_data['ds'], y=forecast_data_orig['yhat_upper'], name='Predicted Upper', mode='lines')
    fig.add_trace(predict_chart_upper)
    predict_chart_lower = go.Scatter(x=forecast_data['ds'], y=forecast_data_orig['yhat_lower'], name='Predicted Lower', mode='lines')
    fig.add_trace(predict_chart_lower)

    # Update layout
    fig.update_layout(title_text='Time Series Forecast', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig)

    # Show the summarized dataframe with forecast values
    st.subheader('Summary of Forecast Data')
    st.write(forecast_data_orig[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Slider for selecting the number of years
n_years = st.slider('Years of prediction:', 1, 4)
forecast_and_plot(n_years)