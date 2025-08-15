# James Fothergill v8255920
# Diabetic-Heart (cardiovascular health monitoring for diabetics)

# Import required libraries for data processing and visualisation

pip install python-dotenv

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pymongo.collection import Collection
import pymongo

load_dotenv()

# Initialise MongoDB connection with caching
@st.cache_resource
def init_mongodb():
    uri = os.getenv("MONGODB_URI")
    client = pymongo.MongoClient(uri)
    db = client["ecg_monitoring"]
    readings_collection = db["readings"]
    ecg_collection = db["ecg_filtered_readings"]
    return readings_collection, ecg_collection


# Cache vital signs data to reduce database queries
@st.cache_data(hash_funcs={Collection: lambda _: None})
def load_vital_signs_data(collection, start_date, end_date):
    # Query data within specified date range
    query = {
        "timestamp": {
            "$gte": start_date,
            "$lte": end_date
        }
    }
    data = list(collection.find(query))
    if not data:
        return pd.DataFrame()
    
    # Process and extract temperature data from nested dictionary
    for doc in data:
        if 'temperature' in doc and isinstance(doc['temperature'], dict):
            doc['target_temp'] = doc['temperature'].get('target_temp')
            doc['ambient_temp'] = doc['temperature'].get('ambient_temp')
    
    return pd.DataFrame(data)

# Cache ECG data to improve loading times
@st.cache_data(hash_funcs={Collection: lambda _: None})
def load_ecg_data(collection, start_date, end_date):
    # Query ECG readings within date range
    query = {
        "timestamp": {
            "$gte": start_date,
            "$lte": end_date
        }
    }
    data = list(collection.find(query))
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)

# Process ECG data to detect abnormalities and calculate metrics
def analyse_ecg_abnormalities(filtered_data):
    try:
        # Convert input data to numpy array for processing
        ecg_data = np.array(filtered_data, dtype=np.float64)
        
        # Ensure data is in correct format (1-dimensional)
        if len(ecg_data.shape) > 1:
            ecg_data = ecg_data.flatten()
        
        # Detect R peaks using signal processing
        peak_height = np.mean(ecg_data) + 0.5 * np.std(ecg_data)
        min_distance = 25  # Minimum distance between peaks (50 Hz sampling rate)
        r_peaks, _ = find_peaks(ecg_data, height=peak_height, distance=min_distance)
        
        # Check if enough peaks were detected for analysis
        if len(r_peaks) < 2:
            return {
                'heart_rate': 0,
                'is_regular': False,
                'r_peaks': None,
                'abnormal_beats': []
            }
        
        # Calculate heart rate from R-R intervals
        sampling_rate = 50  # Hz
        r_intervals = np.diff(r_peaks).astype(np.float64) / sampling_rate
        heart_rate = 60 / np.mean(r_intervals)
        
        # Assess rhythm regularity
        rr_std = np.std(r_intervals)
        is_regular = rr_std < 0.1
        
        # Identify abnormal beats based on amplitude
        abnormal_beats = []
        peak_amplitudes = ecg_data[r_peaks]
        mean_amplitude = np.mean(peak_amplitudes)
        std_amplitude = np.std(peak_amplitudes)
        
        # Mark beats as abnormal if amplitude deviates significantly
        for peak_idx in r_peaks:
            if peak_idx < len(ecg_data):
                peak_amplitude = ecg_data[peak_idx]
                if abs(peak_amplitude - mean_amplitude) > 2 * std_amplitude:
                    abnormal_beats.append(int(peak_idx))
        
        return {
            'heart_rate': float(heart_rate),
            'is_regular': bool(is_regular),
            'r_peaks': r_peaks.tolist(),
            'abnormal_beats': abnormal_beats
        }
    
    except Exception as e:
        # Handle errors and provide debugging information
        st.error(f"Error in ECG analysis: {str(e)}")
        st.error(f"Data type: {type(filtered_data)}")
        st.error(f"Data shape: {filtered_data.shape if hasattr(filtered_data, 'shape') else 'no shape'}")
        return {
            'heart_rate': 0,
            'is_regular': False,
            'r_peaks': None,
            'abnormal_beats': []
        }
# Detect anomalies in vital signs using Isolation Forest algorithm
def detect_anomalies(df):
    # Identify available vital sign features in the dataset
    available_features = []
    if 'heart_rate' in df.columns:
        available_features.append('heart_rate')
    if 'spo2' in df.columns:
        available_features.append('spo2')
    if 'target_temp' in df.columns:
        available_features.append('target_temp')
    
    # Return empty list if no features are available
    if not available_features:
        return []
    
    # Prepare data for anomaly detection
    data = df[available_features].copy()
    
    # Fill missing values with mean to prevent analysis errors
    for col in available_features:
        data[col].fillna(data[col].mean(), inplace=True)
    
    # Initialise and fit Isolation Forest model
    iso_forest = IsolationForest(
        contamination=0.1,  # Expected proportion of anomalies
        random_state=42     # For reproducibility
    )
    
    # Return boolean array where -1 indicates anomalies
    return iso_forest.fit_predict(data) == -1

# Analyse trends in vital signs using exponential smoothing
def analyse_trends(df):
    trend_data = pd.DataFrame()
    available_features = []
    
    # Check which vital signs are available for analysis
    if 'heart_rate' in df.columns:
        available_features.append('heart_rate')
    if 'spo2' in df.columns:
        available_features.append('spo2')
    if 'target_temp' in df.columns:
        available_features.append('target_temp')
    
    # Process each available vital sign
    for feature in available_features:
        # Fill missing values using forward and backward fill
        series = df[feature].fillna(method='ffill').fillna(method='bfill')
        
        if len(series) > 0:
            # Apply exponential smoothing to detect trends
            model = ExponentialSmoothing(
                series,
                trend='add',        # Add trend component
                seasonal=None,      # No seasonality assumed
                seasonal_periods=None
            )
            
            try:
                # Fit model and extract trend
                fitted_model = model.fit()
                trend_data[feature] = fitted_model.fittedvalues
            except Exception as e:
                st.warning(f"Could not analyse trend for {feature}: {str(e)}")
                trend_data[feature] = series
    
    # Set timestamp as index for time series analysis
    if not trend_data.empty:
        trend_data.index = df['timestamp']
    
    return trend_data

# Classify health states based on vital sign thresholds
def classify_health_state(df):
    # Define health state classification rules
    def get_health_state(row):
        state = 'Normal'
        
        # Check heart rate thresholds
        if 'heart_rate' in row:
            hr = float(row['heart_rate'])
            if hr > 100:
                state = 'Severe'      # Tachycardia
            elif hr > 90:
                state = 'Warning'     # Elevated heart rate
            elif hr > 75:
                state = 'Caution'     # Slightly elevated
            elif hr < 50:
                state = 'Severe'      # Bradycardia
            elif hr < 60:
                state = 'Warning'     # Low heart rate
        
        # Check SpO2 thresholds
        if 'spo2' in row:
            spo2 = float(row['spo2'])
            if spo2 < 90:
                state = 'Severe'      # Severe hypoxaemia
            elif spo2 < 92:
                state = 'Warning'     # Moderate hypoxaemia
            elif spo2 < 95:
                state = 'Caution'     # Mild hypoxaemia
        
        # Check temperature thresholds
        if 'target_temp' in row and row['target_temp'] is not None:
            temp = float(row['target_temp'])
            if temp > 38.5:
                state = 'Severe'      # High fever
            elif temp > 38.0:
                state = 'Warning'     # Fever
            elif temp > 37.5:
                state = 'Caution'     # Elevated temperature
        
        return state
    
    # Identify available vital signs for classification
    available_features = []
    if 'heart_rate' in df.columns:
        available_features.append('heart_rate')
    if 'spo2' in df.columns:
        available_features.append('spo2')
    if 'target_temp' in df.columns:
        available_features.append('target_temp')
    
    # Return empty results if no features available
    if not available_features:
        return [], []
    
    # Prepare data for classification
    X = df[available_features].copy()
    
    # Fill missing values with mean
    for col in available_features:
        X[col].fillna(X[col].mean(), inplace=True)
    
    # Generate health state labels
    y = df.apply(get_health_state, axis=1)
    
    # Train Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'  # Account for class imbalance
    )
    clf.fit(X, y)
    
    # Generate predictions and probabilities
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)
    
    return predictions, probabilities

# Main application function
def main():
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="Diabetic Cardiovascular Health Monitor",
        page_icon="❤️",
        layout="wide"
    )
    
    # Display primary medical disclaimer at the top of the app
    st.error("""
    **⚠️ IMPORTANT MEDICAL DISCLAIMER**

    This tool is NOT a medical diagnostic device:
    * It is designed for monitoring and educational purposes only
    * It does NOT replace professional medical diagnosis or treatment
    * The analysis provided is based on algorithms and statistical patterns
    * Results should ALWAYS be verified by qualified healthcare professionals
    * No medical decisions should be made based solely on this tool
    * Emergency situations require immediate medical attention
    
    If you experience any concerning symptoms, contact your healthcare provider 
    or emergency services immediately.
    """)
    
    st.title("Diabetic Cardiovascular Health Monitor")
    
    # Initialise MongoDB connections
    readings_collection, ecg_collection = init_mongodb()

    try:
        # Sidebar date range selector
        st.sidebar.header("Date Range Selection")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        min_date = datetime(2025, 1, 1).date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(start_date.date(), end_date.date()),
            min_value=min_date,
            max_value=end_date.date()
        )
        
        if len(date_range) == 2:
            # Convert selected dates to datetime
            start_date, end_date = date_range
            start_date = datetime.combine(start_date, datetime.min.time())
            end_date = datetime.combine(end_date, datetime.max.time())
            
            # Load data from MongoDB
            vital_signs_df = load_vital_signs_data(readings_collection, start_date, end_date)
            ecg_df = load_ecg_data(ecg_collection, start_date, end_date)
            
            # Check if data is available
            if vital_signs_df.empty:
                st.warning("No vital signs data found for the selected date range.")
                return
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Vital Signs Overview", 
                "Anomaly Detection", 
                "Trend Analysis",
                "Health State Classification",
                "ECG Analysis",
                "Information Guide"
            ])
            
            # Vital Signs Overview Tab
            with tab1:
                st.header("Vital Signs Overview")
                st.info("""
                **Data Sources and Measurements:**
                * Heart Rate (bpm)
                * Blood Oxygen/SpO2 (%)
                * Body Temperature (°C)
                
                **Visualisation Method:**
                * Current values: Latest reading from database
                * Time series plots: Raw data over selected time period
                * Direct measurements - no algorithms applied
                
                **Refresh Rate:**
                * Data updates every time the page loads
                * Latest values shown at the top
                * Historical trends shown in graphs below
                """)
                
                st.subheader("Current Status")
                latest = vital_signs_df.iloc[-1]
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'heart_rate' in latest:
                        st.metric("Heart Rate", f"{latest.get('heart_rate', 'N/A')} bpm")
                with col2:
                    if 'spo2' in latest:
                        st.metric("SpO2", f"{latest.get('spo2', 'N/A')}%")
                with col3:
                    if 'target_temp' in latest:
                        st.metric("Temperature", f"{latest.get('target_temp', 'N/A')}°C")
                
                # Plot heart rate time series
                if 'heart_rate' in vital_signs_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=vital_signs_df['timestamp'],
                        y=vital_signs_df['heart_rate'],
                        name='Heart Rate'
                    ))
                    fig.update_layout(
                        title="Heart Rate Over Time",
                        yaxis_title="BPM",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig)
                
                # Plot SpO2 time series
                if 'spo2' in vital_signs_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=vital_signs_df['timestamp'],
                        y=vital_signs_df['spo2'],
                        name='SpO2'
                    ))
                    fig.update_layout(
                        title="SpO2 Over Time",
                        yaxis_title="%",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig)
                
                # Plot temperature time series
                if 'target_temp' in vital_signs_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=vital_signs_df['timestamp'],
                        y=vital_signs_df['target_temp'],
                        name='Temperature'
                    ))
                    fig.update_layout(
                        title="Temperature Over Time",
                        yaxis_title="°C",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig)

            # Anomaly Detection Tab
            with tab2:
                st.header("Anomaly Detection")
                st.info("""
                **Algorithm Used: Isolation Forest**
                * An unsupervised machine learning algorithm that detects outliers
                * Contamination factor: 0.1 (expects 10% of data points may be anomalous)
                
                **Data Sources Used:**
                * Heart Rate (bpm)
                * SpO2 (%)
                * Temperature (°C)
                
                **How the Analysis Works:**
                1. Combines all available vital signs into a single dataset
                2. Fills any missing values with the mean of that vital sign
                3. Identifies points that deviate significantly from normal patterns
                4. Visualises anomalies in red on the graph
                
                **Interpretation:**
                * Blue dots: Normal readings
                * Red dots: Potential anomalies
                * Multiple red dots in sequence may indicate a significant event
                """)
                
                anomalies = detect_anomalies(vital_signs_df)
                
                if len(anomalies) > 0:
                    fig = go.Figure()
                    
                    # Plot normal readings
                    normal_mask = ~anomalies
                    fig.add_trace(go.Scatter(
                        x=vital_signs_df[normal_mask]['timestamp'],
                        y=vital_signs_df[normal_mask]['heart_rate'],
                        name='Normal',
                        mode='markers',
                        marker=dict(color='blue', size=6)
                    ))
                    
                    # Plot anomalous readings
                    anomaly_mask = anomalies
                    fig.add_trace(go.Scatter(
                        x=vital_signs_df[anomaly_mask]['timestamp'],
                        y=vital_signs_df[anomaly_mask]['heart_rate'],
                        name='Anomaly',
                        mode='markers',
                        marker=dict(color='red', size=10)
                    ))
                    
                    fig.update_layout(
                        title="Heart Rate Anomalies",
                        yaxis_title="BPM",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig)
                    
                    st.write(f"Detected {sum(anomalies)} anomalies in {len(anomalies)} readings")
            
            # Trend Analysis Tab
            with tab3:
                st.header("Trend Analysis")
                st.info("""
                **Algorithm Used: Exponential Smoothing**
                * Time series forecasting method
                * Applies greater weight to recent observations
                * Helps identify underlying trends
                
                **Data Sources Used:**
                * Heart Rate (bpm)
                * SpO2 (%)
                * Temperature (°C)
                
                **How the Analysis Works:**
                1. Processes each vital sign independently
                2. Fills missing values using forward/backward fill
                3. Applies exponential smoothing algorithm
                4. Generates smoothed trend line
                
                **Visualisation Guide:**
                * Blue line: Actual measurements
                * Red dashed line: Smoothed trend
                * Divergence between lines indicates variability
                """)
                
                trend_data = analyse_trends(vital_signs_df)
                
                if not trend_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=vital_signs_df['timestamp'],
                        y=vital_signs_df['heart_rate'],
                        name='Actual Heart Rate',
                        mode='lines+markers',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=trend_data.index,
                        y=trend_data['heart_rate'],
                        name='Trend',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title="Heart Rate Trend Analysis",
                        yaxis_title="BPM",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig)
            
            # Health State Classification Tab
            with tab4:
                st.header("Health State Classification")
                st.info("""
                **Algorithm Used: Random Forest Classifier**
                * Supervised machine learning with balanced class weights
                * Ensemble of 100 decision trees
                
                **Data Sources and Thresholds:**
                
                Heart Rate:
                * Severe: >100 bpm or <50 bpm
                * Warning: 90-100 bpm or 50-60 bpm
                * Caution: 75-90 bpm
                * Normal: 60-75 bpm
                
                SpO2:
                * Severe: <90%
                * Warning: 90-92%
                * Caution: 92-95%
                * Normal: ≥95%
                
                Temperature:
                * Severe: >38.5°C
                * Warning: >38.0°C
                * Caution: >37.5°C
                * Normal: ≤37.5°C
                
                **Colour Coding:**
                * Normal: Blue
                * Caution: Light Blue
                * Warning: Orange
                * Severe: Red
                
                **Classification Process:**
                1. Evaluates each vital sign against defined thresholds
                2. Takes the most severe state among all measurements
                3. Trains classifier on labelled data
                4. Generates probability scores for each state
                """)

                # Health State Classification visualisation
                predictions, probabilities = classify_health_state(vital_signs_df)
                
                # Plot health states with updated colours
                fig = go.Figure()
                colours = {
                    'Normal': 'blue',      # Normal state in blue
                    'Caution': 'lightblue', # Caution state in light blue
                    'Warning': 'orange',    # Warning state in orange
                    'Severe': 'red'         # Severe state in red
                }
                
                for state in colours:
                    mask = predictions == state
                    if any(mask):
                        fig.add_trace(go.Scatter(
                            x=vital_signs_df[mask]['timestamp'],
                            y=vital_signs_df[mask]['heart_rate'],
                            name=state,
                            mode='markers',
                            marker=dict(color=colours[state], size=8)
                        ))
                
                fig.update_layout(
                    title="Health State Classification",
                    yaxis_title="Heart Rate (BPM)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig)
                
                # Plot health state distribution
                state_counts = pd.Series(predictions).value_counts()
                fig = go.Figure(data=[go.Bar(
                    x=state_counts.index,
                    y=state_counts.values,
                    marker_color=[colours[state] for state in state_counts.index]
                )])
                fig.update_layout(
                    title="Health State Distribution",
                    xaxis_title="Health State",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig)
            
            # ECG Analysis Tab
            with tab5:
                st.header("ECG Analysis")
                st.info("""
                **Algorithms Used:**
                1. Peak Detection: scipy.signal.find_peaks
                2. Statistical Analysis for Beat Classification
                
                **Data Sources:**
                * Filtered ECG signal (50 Hz sampling rate)
                * Time window selected by user
                
                **Analysis Process:**
                1. Peak Detection:
                   * Threshold: mean + 0.5 × standard deviation
                   * Minimum distance between peaks: 25 samples (0.5 seconds)
                
                2. Heart Rate Calculation:
                   * Based on R-R intervals
                   * Formula: 60 ÷ mean(R-R intervals)
                
                3. Rhythm Regularity:
                   * Based on standard deviation of R-R intervals
                   * Regular if standard deviation < 0.1 seconds
                
                4. Abnormal Beat Detection:
                   * Mean ± 2 standard deviations of peak amplitudes
                   * Marks outliers as abnormal beats
                
                **Visualisation Guide:**
                * Blue line: ECG signal
                * Green dots: R peaks (normal beats)
                * Red X: Abnormal beats
                """)
                
                if not ecg_df.empty:
                    st.subheader("ECG Statistics")
                    total_readings = len(ecg_df)
                    time_span = (ecg_df['timestamp'].max() - ecg_df['timestamp'].min()).total_seconds()
                    st.write(f"Total readings: {total_readings}")
                    st.write(f"Time span: {time_span:.2f} seconds")
                    
                    # Time window selector
                    window_seconds = st.slider(
                        "Select time window (seconds)",
                        min_value=1,
                        max_value=min(60, int(time_span)),
                        value=10
                    )
                    
                    # Get the last window_seconds of data
                    latest_data = ecg_df.iloc[-int(window_seconds * 5):]  # 5 documents per second
                    
                    if not latest_data.empty and 'filtered_data' in latest_data.columns:
                        try:
                            # Concatenate filtered_data arrays
                            all_data = np.concatenate(latest_data['filtered_data'].values)
                            x_values = np.arange(len(all_data))
                            
                            # Analyse ECG data
                            analysis_results = analyse_ecg_abnormalities(all_data)
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Heart Rate", f"{analysis_results['heart_rate']:.0f} bpm")
                            with col2:
                                st.metric("Rhythm Regularity", 
                                         "Regular" if analysis_results['is_regular'] else "Irregular")
                            with col3:
                                st.metric("Abnormal Beats", 
                                         f"{len(analysis_results['abnormal_beats'])}")
                            
                            # Plot ECG data
                            fig = go.Figure()
                            
                            # Plot filtered ECG data
                            fig.add_trace(go.Scatter(
                                x=x_values,
                                y=all_data,
                                name='Filtered ECG',
                                line=dict(color='blue', width=1)
                            ))
                            
                            # Plot R peaks
                            if analysis_results['r_peaks'] is not None and len(analysis_results['r_peaks']) > 0:
                                r_peak_x = analysis_results['r_peaks']
                                r_peak_y = [all_data[i] for i in r_peak_x if i < len(all_data)]
                                if len(r_peak_x) == len(r_peak_y):
                                    fig.add_trace(go.Scatter(
                                        x=r_peak_x,
                                        y=r_peak_y,
                                        mode='markers',
                                        name='R Peaks',
                                        marker=dict(color='green', size=8, symbol='circle')
                                    ))

                            # Plot abnormal beats
                            if analysis_results['abnormal_beats'] and len(analysis_results['abnormal_beats']) > 0:
                                abnormal_x = analysis_results['abnormal_beats']
                                abnormal_y = [all_data[i] for i in abnormal_x if i < len(all_data)]
                                if len(abnormal_x) == len(abnormal_y):
                                    fig.add_trace(go.Scatter(
                                        x=abnormal_x,
                                        y=abnormal_y,
                                        mode='markers',
                                        name='Abnormal Beats',
                                        marker=dict(color='red', size=10, symbol='x')
                                    ))
                            
                            # Update layout
                            fig.update_layout(
                                title=f"ECG Data (Last {window_seconds} seconds)",
                                yaxis_title="Amplitude",
                                xaxis_title="Sample",
                                hovermode='x unified',
                                showlegend=True,
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
                            )
                            st.plotly_chart(fig)
                            
                            # Download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Download Selected Window"):
                                    csv = latest_data.to_csv(index=False)
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name=f"ecg_data_{window_seconds}s.csv",
                                        mime="text/csv"
                                    )
                            with col2:
                                if st.button("Download All Data"):
                                    csv = ecg_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name="ecg_data_complete.csv",
                                        mime="text/csv"
                                    )
                        
                        except Exception as e:
                            st.error(f"Error analysing ECG data: {str(e)}")
                            st.error("Debug information:")
                            st.error(f"Data type: {type(latest_data['filtered_data'].iloc[0])}")
                            st.error(f"Data sample: {latest_data['filtered_data'].iloc[0][:10]}")
                    else:
                        st.warning("No ECG data available for the selected time window")
                else:
                    st.warning("No ECG data available")

            # Information Guide Tab
            with tab6:
                st.header("Health Monitoring Information Guide")
                
                # Primary Medical Disclaimer
                st.error("""
                **⚠️ IMPORTANT: NOT A DIAGNOSTIC TOOL**
                
                This monitoring system:
                * Is for educational and monitoring purposes only
                * Does NOT replace medical professional judgement
                * Should NOT be used for self-diagnosis
                * Requires interpretation by healthcare professionals
                """)
                
                # Vital Signs Guide
                st.subheader("📊 Understanding Vital Signs")
                st.markdown("""
                **Heart Rate (Normal: 60-100 bpm)**
                * Below 60 bpm: Bradycardia
                * Above 100 bpm: Tachycardia
                * Irregular rhythm may indicate arrhythmia
                
                **Blood Oxygen/SpO2 (Normal: 95-100%)**
                * 92-95%: Mild hypoxaemia
                * 89-92%: Moderate hypoxaemia
                * Below 89%: Severe hypoxaemia
                
                **Body Temperature**
                * Normal: 36.5-37.5°C
                * Fever: Above 38.0°C
                * High Fever: Above 38.5°C
                """)
                
                # ECG Guide
                st.subheader("💓 Understanding ECG Analysis")
                st.markdown("""
                **R Peaks**
                * Represent ventricular contraction
                * Regular spacing indicates normal rhythm
                * Irregular spacing may indicate arrhythmia
                
                **Abnormal Beats**
                * Unusual amplitude or timing
                * May indicate ectopic beats or conduction issues
                * Multiple abnormal beats warrant medical attention
                
                **Rhythm Analysis**
                * Regular: Consistent R-R intervals
                * Irregular: Variable R-R intervals
                * Pattern changes may indicate cardiac conditions
                """)
                
                # Health States Guide
                st.subheader("🏥 Health State Classifications")
                st.markdown("""
                **Normal (Blue)**
                * All vital signs within normal ranges
                * Regular heart rhythm
                * Normal ECG patterns
                
                **Caution (Light Blue)**
                * Slight deviations from normal ranges
                * Minor rhythm irregularities
                * May require monitoring
                
                **Warning (Orange)**
                * Significant deviations from normal
                * Multiple irregular beats
                * Requires medical attention
                
                **Severe (Red)**
                * Critical deviations from normal
                * Sustained abnormal patterns
                * Requires immediate medical attention
                """)
                
                # Emergency Warning Signs
                st.error("""
                **🚨 Seek Immediate Medical Care If:**
                * Chest pain or pressure
                * Severe shortness of breath
                * Fainting or severe dizziness
                * Sustained irregular heartbeat
                * SpO2 below 88%
                * Severe headache with high blood pressure
                """)
                
                # Monitoring Best Practices
                st.info("""
                **📋 Best Practices for Monitoring:**
                * Check vital signs at consistent times
                * Record any symptoms with abnormal readings
                * Keep a log of medication changes
                * Share monitoring data with healthcare providers
                * Ensure proper sensor placement for accurate readings
                * Report technical issues to your healthcare team
                """)

    except Exception as e:
        st.error(f"Application error: {str(e)}")

# Run the application
if __name__ == "__main__":
    main()
