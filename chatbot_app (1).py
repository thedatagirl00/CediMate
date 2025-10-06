
import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import numpy as np
import logging
import os
from datetime import datetime

# --- Configuration and Setup ---
# Set up logging
logging.basicConfig(filename='chatbot_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Data file path (assuming it's in the same directory)
DATA_FILE = 'merged_exchange_rates.csv'

# Retraining configuration
RETRAIN_INTERVAL_DAYS = 30 # Retrain every 30 days
LAST_RETRAIN_FILE = 'last_retrain.txt'

# Define KPIs
# Use Streamlit's session state for KPIs to persist across reruns
if 'kpis' not in st.session_state:
    st.session_state['kpis'] = {
        'arima_mae': None,
        'arima_mse': None,
        'arima_rmse': None,
        'chatbot_response_time': [], # Store response times to calculate average
        'successful_requests': 0,
        'failed_requests': 0
    }
KPIs = st.session_state['kpis']


# --- Data Loading and Preprocessing ---
# Function to load and preprocess data
@st.cache_data(ttl=RETRAIN_INTERVAL_DAYS*24*60*60) # Cache data for the retraining interval
def load_and_preprocess_data():
    '''Loads and preprocesses the exchange rate data.'''
    try:
        df_merged = pd.read_csv(DATA_FILE)
        logging.info(f"Successfully loaded data from {DATA_FILE}")

        month_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df_variable = df_merged.melt(id_vars=['Year', 'Variables'], value_vars=month_columns, var_name='Month', value_name='Exchange_Rate')
        df_variable['Date'] = pd.to_datetime(df_variable['Year'].astype(str) + '-' + df_variable['Month'], format='%Y-%b')
        df_variable = df_variable.sort_values(['Variables', 'Date']).reset_index(drop=True)

        # Select a specific exchange rate time series
        selected_variable_chatbot = 'Inter-Bank Exchange Rate - End Period (GHC/US$)'
        df_selected_chatbot = df_variable[df_variable['Variables'] == selected_variable_chatbot].copy()
        df_selected_chatbot = df_selected_chatbot.sort_values('Date').reset_index(drop=True)

        # Drop rows with NaN values in the Exchange_Rate column
        df_selected_chatbot.dropna(subset=['Exchange_Rate'], inplace=True)
        logging.info("Data preprocessing complete.")
        return df_selected_chatbot
    except FileNotFoundError:
        logging.error(f"Data file not found: {DATA_FILE}")
        st.error(f"Error: '{DATA_FILE}' not found. Please ensure the merged data file is in the same directory.")
        st.stop()
    except Exception as e:
        logging.error(f"An error occurred during data loading or preprocessing: {e}")
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        st.stop()

# --- Model Training and Management ---
# Function to train ARIMA model
def train_arima_model(data):
    '''Trains the ARIMA model.'''
    try:
        p, d, q = 5, 1, 0 # Example orders
        model = ARIMA(data['Exchange_Rate'].reset_index(drop=True), order=(p, d, q))
        model_fit = model.fit()
        logging.info("ARIMA model trained successfully.")
        return model_fit
    except Exception as e:
        logging.error(f"An error occurred during ARIMA model training: {e}")
        return None

# Function to train GARCH model
def train_garch_model(data):
    '''Trains the GARCH model.'''
    try:
        returns = 100 * data['Exchange_Rate'].pct_change().dropna()
        if returns.empty:
            logging.warning("Insufficient data to train GARCH model.")
            return None

        garch_model = arch_model(returns, vol='Garch', p=1, q=1)
        garch_results = garch_model.fit(disp='off')
        logging.info("GARCH model trained successfully.")
        return garch_results
    except Exception as e:
        logging.error(f"An error occurred during GARCH model training: {e}")
        return None

# Function to check if retraining is needed
def needs_retraining():
    '''Checks if the models need to be retrained based on the interval.'''
    if not os.path.exists(LAST_RETRAIN_FILE):
        return True
    try:
        with open(LAST_RETRAIN_FILE, 'r') as f:
            last_retrain_str = f.read()
        last_retrain_date = datetime.strptime(last_retrain_str, '%Y-%m-%d')
        time_since_last_retrain = datetime.now() - last_retrain_date
        return time_since_last_retrain.days >= RETRAIN_INTERVAL_DAYS
    except Exception as e:
        logging.error(f"Error checking last retrain date: {e}")
        return True # Retrain in case of error

# Function to update last retrain date
def update_last_retrain_date():
    '''Updates the file with the current date as the last retrain date.'''
    try:
        with open(LAST_RETRAIN_FILE, 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d'))
        logging.info("Last retrain date updated.")
    except Exception as e:
        logging.error(f"Error updating last retrain date file: {e}")

# --- Chatbot Logic ---
# Function to handle chatbot responses
def chatbot_response(user_input, arima_model, garch_model, data):
    '''Processes user input and returns model results.'''
    user_input_lower = user_input.lower()
    response = "I didn't understand your request. Please ask for 'forecast for X months' or 'analyze volatility for X months'."
    response_status = "Failed" # Track response status

    if "forecast" in user_input_lower:
        if arima_model is None:
            response = "ARIMA model is not available. Please try again later."
            logging.warning("Forecast requested but ARIMA model is None.")
        else:
            try:
                horizon_str = ''.join(filter(str.isdigit, user_input_lower.split("forecast")[-1]))
                forecast_horizon = int(horizon_str) if horizon_str else 12

                if forecast_horizon <= 0:
                     response = "Please provide a positive number of months for the forecast horizon."
                else:
                    logging.info(f"Processing forecast request for {forecast_horizon} months.")
                    # Use the trained ARIMA model
                    future_predictions = arima_model.predict(start=len(data), end=len(data) + forecast_horizon - 1)

                    last_date = data['Date'].iloc[-1]
                    future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='MS')[1:]
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Exchange_Rate': future_predictions.values})

                    response = "Forecast Results:\n" + forecast_df.to_string(index=False)
                    response_status = "Successful"
                    logging.info("Forecast request processed successfully.")

            except ValueError:
                response = "Please specify the number of months you want to forecast (e.g., 'forecast for 12 months')."
                logging.warning(f"Invalid forecast horizon in input: {user_input}")
            except Exception as e:
                response = f"An error occurred during forecasting: {e}"
                logging.error(f"Error during forecasting: {e}")

    elif "volatility" in user_input_lower:
        if garch_model is None:
            response = "GARCH model is not available. Please try again later."
            logging.warning("Volatility requested but GARCH model is None.")
        else:
            try:
                horizon_str = ''.join(filter(str.isdigit, user_input_lower.split("volatility")[-1]))
                forecast_horizon = int(horizon_str) if horizon_str else 12

                if forecast_horizon <= 0:
                     response = "Please provide a positive number of months for the volatility forecast horizon."
                else:
                    logging.info(f"Processing volatility analysis request for {forecast_horizon} months.")
                    # Use the trained GARCH model
                    volatility_forecasts = garch_model.forecast(horizon=forecast_horizon)
                    forecasted_variance = volatility_forecasts.variance.dropna().tail(forecast_horizon)

                    if forecasted_variance.empty:
                        response = "Could not generate volatility forecasts."
                        logging.warning("GARCH volatility forecasts are empty.")
                    else:
                        last_date = data['Date'].iloc[-1]
                        future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='MS')[1:]
                        if len(future_dates) > len(forecasted_variance):
                            future_dates = future_dates[:len(forecasted_variance)]
                        elif len(future_dates) < len(forecasted_variance):
                             forecasted_variance = forecasted_variance.iloc[:len(future_dates)]

                        volatility_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Conditional_Volatility': forecasted_variance.values.flatten()})
                        response = "Volatility Analysis Results:\n" + volatility_df.to_string(index=False)
                        response_status = "Successful"
                        logging.info("Volatility request processed successfully.")

            except ValueError:
                response = "Please specify the number of months you want to analyze volatility for (e.g., 'analyze volatility for 12 months')."
                logging.warning(f"Invalid volatility horizon in input: {user_input}")
            except Exception as e:
                response = f"An error occurred during volatility analysis: {e}"
                logging.error(f"Error during volatility analysis: {e}")
    else:
        logging.info(f"Unrecognized user input: {user_input}")

    # Update KPIs based on response status
    if response_status == "Successful":
        st.session_state['kpis']['successful_requests'] += 1
    else:
        st.session_state['kpis']['failed_requests'] += 1

    return response

# --- Streamlit App Layout ---
st.title("CediMate: GHC/US$ Exchange Rate Chatbot")
st.write("Hello! I can provide exchange rate forecasts and volatility analysis for the GHC/US$.")
st.write("You can ask for 'forecast for X months' or 'analyze volatility for X months', replacing X with the number of months.")

# Load and preprocess data
df_selected_chatbot = load_and_preprocess_data()

# Model training/loading logic
arima_model = None
garch_model = None

# In a real application, you would save and load models to avoid retraining
# every time the app reruns unless retraining is due.
# For this example, we'll retrain if needed based on the timestamp,
# but still train on every rerun if not due, which is inefficient for production.
# A better approach is to save/load models.

if needs_retraining():
    st.info("Retraining models...")
    logging.info("Initiating model retraining.")
    arima_model = train_arima_model(df_selected_chatbot)
    garch_model = train_garch_model(df_selected_chatbot)
    update_last_retrain_date()
    st.success("Models retrained successfully.")
    logging.info("Model retraining completed.")
else:
    st.info("Models are up-to-date.")
    logging.info("Models up-to-date. Training models for the current session (for demo purposes).")
    # In a production app, you'd load saved models here
    arima_model = train_arima_model(df_selected_chatbot) # Retrain for demo simplicity
    garch_model = train_garch_model(df_selected_chatbot) # Retrain for demo simplicity


user_input = st.text_input("Enter your request here:", "")

if user_input:
    # start_time = datetime.now() # For response time tracking
    response = chatbot_response(user_input, arima_model, garch_model, df_selected_chatbot)
    st.write(response)
    # end_time = datetime.now()
    # duration = (end_time - start_time).total_seconds()
    # st.session_state['kpis']['chatbot_response_time'].append(duration) # Append response time


# Display KPIs
st.sidebar.title("Chatbot Performance Metrics")
st.sidebar.write(f"Successful Requests: {st.session_state['kpis']['successful_requests']}")
st.sidebar.write(f"Failed Requests: {st.session_state['kpis']['failed_requests']}")

# Calculate and display average response time (if tracked)
# if st.session_state['kpis']['chatbot_response_time']:
#     avg_response_time = np.mean(st.session_state['kpis']['chatbot_response_time'])
#     st.sidebar.write(f"Average Response Time: {avg_response_time:.2f} seconds")
# else:
#     st.sidebar.write("Average Response Time: N/A")


# Note: More sophisticated KPI tracking (like MAE/RMSE over time,
# detailed response time, etc.) would require storing data and
# implementing a separate monitoring dashboard. This is a basic
# in-app display using session state (cleared on app restart).

# Mechanism for collecting new data: This is an external process.
# The application assumes the 'merged_exchange_rates.csv' file
# is updated periodically with new data. A separate script would
# be needed to fetch, clean, merge, and save new data to this file.
# The `st.cache_data` with ttl helps in picking up new data
# after the cache expires (every RETRAIN_INTERVAL_DAYS).

# Handling potential issues:
# - Data quality: Handled partially by dropping NaNs. More robust
#   validation is needed in a production system.
# - Model drift: Handled by periodic retraining. More advanced
#   monitoring of model performance on validation data is recommended.
# - Increased user traffic: Requires scalable deployment infrastructure
#   (e.g., Streamlit Cloud, cloud platforms).
# - Error Handling: Basic try-except blocks are included. More specific
#   error handling and user-friendly messages are needed.
# - Manual Trigger for Retraining: A simple button could be added
#   that, when clicked, updates the last_retrain.txt file to force
#   retraining on the next app rerun (or use a different mechanism
#   like a dedicated endpoint if deployed).
# Example of a manual retraining trigger (uncomment and adapt if needed):
# if st.sidebar.button("Force Retrain"):
#     update_last_retrain_date()
#     st.sidebar.success("Forced retraining. Please refresh the app.") # Or rerun the app logic


