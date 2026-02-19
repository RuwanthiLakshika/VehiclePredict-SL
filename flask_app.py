"""
Vehicle Registration Predictor - Flask Web Application

Interactive web app for Sri Lankan vehicle registration forecasting.
Features:
- Monthly registration predictions by category
- SHAP-based explainability
- Data visualization and trend analysis
- Economic impact assessment

Model: Bayesian-Optimized CatBoost with Log-Transform (R² = 0.9842, MAE = 160.10 vehicles/month)
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import os
import io
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ML imports
from catboost import CatBoostRegressor
import shap
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global variables for caching
MODEL = None
DATA = None
EXPLAINER = None
FEATURE_ENCODERS = {}


def load_model():
    """Load the trained CatBoost model."""
    global MODEL
    if MODEL is None:
        model_path = 'models/catboost_model.bin'
        if not os.path.exists(model_path):
            return None
        MODEL = CatBoostRegressor()
        MODEL.load_model(model_path)
    return MODEL


def load_data():
    """Load the dataset."""
    global DATA
    if DATA is None:
        data_path = 'data/master_dataset.csv'
        if not os.path.exists(data_path):
            return None
        DATA = pd.read_csv(data_path)
    return DATA


def get_explainer():
    """Get SHAP explainer."""
    global EXPLAINER
    if EXPLAINER is None:
        model = load_model()
        data = load_data()
        if model is not None and data is not None:
            # Use sample for explainer
            # Prepare features for SHAP explainer
            exclude_cols = ['Standard_Category', 'Month', 'Year', 'New_Registration']
            X = data.drop(columns=exclude_cols, errors='ignore')
            # Select numeric columns only for SHAP
            X_numeric = X.select_dtypes(include=[np.number])
            EXPLAINER = shap.TreeExplainer(model)
    return EXPLAINER


def prepare_features(data_dict):
    """Prepare features for vehicle registration prediction."""
    try:
        # Define expected features used during training
        feature_cols = ['Transfer', 'Yearly_Total_Stock', 'Month_Num', 'Quarter', 
                       'Is_Peak_Season', 'Is_Crisis_Period', 'Transfer_to_New_Ratio',
                       'New_Registration_Market_Share', 'Prev_Month_New_Reg', 'Monthly_Growth_Rate']
        
        # Create DataFrame with input data
        df_input = pd.DataFrame([data_dict])
        
        # Ensure all required features are present
        for col in feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0.0
        
        # Select only the required features in correct order
        return df_input[feature_cols]
    except Exception as e:
        print(f"Feature preparation error: {e}")
        return None


@app.route('/')
def index():
    """Home page."""
    model = load_model()
    data = load_data()
    
    stats = {}
    if data is not None:
        if 'New_Registration' in data.columns:
            stats = {
                'total_months': len(data),
                'avg_registrations': f"{data['New_Registration'].mean():.0f}",
                'registration_range': f"{data['New_Registration'].min():.0f} - {data['New_Registration'].max():.0f}",
                'vehicle_categories': len(data['Standard_Category'].unique()) if 'Standard_Category' in data.columns else 0
            }
        else:
            stats = {'error': 'Data structure not recognized'}
    
    return render_template('index.html', stats=stats, model_loaded=model is not None)


@app.route('/explainability')
def explainability():
    """SHAP Explainability page."""
    return render_template('explainability.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page and API endpoint."""
    if request.method == 'GET':
        # Load data for form options
        data = load_data()
        
        categories = []
        
        if data is not None:
            if 'Standard_Category' in data.columns:
                categories = sorted(data['Standard_Category'].unique().tolist())
        
        # Note: Year input is now free-form (2026 onwards) for future predictions
        # Months are hardcoded in template for all 12 months
        return render_template('predict.html', 
                             categories=categories,
                             months=[],  # Not needed - all months in template
                             years=[])   # Not needed - year is number input field
    
    elif request.method == 'POST':
        """Make prediction for future time periods."""
        try:
            data_dict = request.get_json()
            model = load_model()
            
            if model is None:
                return jsonify({'error': 'Model not loaded. Please train the model first.'}), 400
            
            # Prepare features
            X = prepare_features(data_dict)
            if X is None:
                return jsonify({'error': 'Invalid input data'}), 400
            
            # Make prediction (model outputs log-scale, inverse transform)
            prediction_log = model.predict(X)[0]
            prediction = max(float(np.expm1(prediction_log)), 0)
            
            # Get feature importance for this prediction
            explainer = get_explainer()
            importance = {}
            
            if explainer is not None:
                try:
                    shap_values = explainer.shap_values(X)
                    feature_names = X.columns.tolist()
                    
                    # Handle different SHAP output shapes
                    # shap_values can be: list, 2D array (1, n_features), or 1D array
                    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                        sv = shap_values[0]  # First (only) sample
                    elif isinstance(shap_values, list):
                        sv = shap_values[0]
                    else:
                        sv = shap_values
                    
                    # Human-readable feature labels
                    label_map = {
                        'Transfer': 'Vehicle Transfers',
                        'Yearly_Total_Stock': 'Total Fleet Size',
                        'Month_Num': 'Month of Year',
                        'Quarter': 'Quarter',
                        'Is_Peak_Season': 'Peak Season',
                        'Is_Crisis_Period': 'Crisis Period',
                        'Transfer_to_New_Ratio': 'Transfer-to-New Ratio',
                        'New_Registration_Market_Share': 'Market Share',
                        'Prev_Month_New_Reg': 'Previous Month Registrations',
                        'Monthly_Growth_Rate': 'Monthly Growth Rate'
                    }
                    
                    # Get feature importance from SHAP values
                    for i, feat in enumerate(feature_names):
                        label = label_map.get(feat, feat)
                        importance[label] = float(np.abs(sv[i]))
                    
                    # Sort and get top 5
                    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
                except Exception as e:
                    print(f"SHAP error: {e}")
                    import traceback
                    traceback.print_exc()
                    importance = {}
            
            return jsonify({
                'success': True,
                'prediction': round(float(prediction), 0),
                'prediction_text': f"{round(float(prediction), 0):.0f} vehicles",
                'confidence': 'High (±160 vehicles, ~9% median error)',
                'model_r2': 0.9842,
                'importance': importance,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 400


@app.route('/api/explore')
def explore_api():
    """Get data for exploration."""
    try:
        data = load_data()
        if data is None:
            return jsonify({'error': 'Data not loaded'}), 400
        
        # Prepare summary statistics with safe conversions
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        numeric_summary = {}
        if numeric_cols:
            summary_df = data[numeric_cols].describe()
            numeric_summary = summary_df.fillna(0).to_dict()
        
        summary = {
            'shape': list(data.shape),
            'columns': data.columns.tolist(),
            'dtypes': data.dtypes.astype(str).to_dict(),
            'numeric_summary': numeric_summary
        }
        
        # Prepare data sample - convert all to strings/numbers to avoid serialization issues
        data_sample = []
        for _, row in data.iterrows():  # Get ALL rows instead of just first 10
            sample_row = {}
            for col in data.columns:  # Include all columns
                val = row[col]
                if pd.isna(val):
                    sample_row[col] = 'N/A'
                elif isinstance(val, (int, float)):
                    sample_row[col] = round(float(val), 2)
                else:
                    sample_row[col] = str(val)
            data_sample.append(sample_row)
        
        return jsonify({
            'summary': summary,
            'data_sample': data_sample
        })
    
    except Exception as e:
        print(f"Explore error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error loading data: {str(e)}'}), 400


@app.route('/api/chart/<chart_type>')
def get_chart(chart_type):
    """Get chart data."""
    try:
        data = load_data()
        if data is None:
            return jsonify({'error': 'Data not loaded'}), 400
        
        elif chart_type == 'registration_distribution':
            # Show distribution of monthly registrations
            if 'New_Registration' not in data.columns:
                return jsonify({'error': 'New_Registration column not found'}), 400
            
            registrations = data['New_Registration'].dropna().tolist()
            
            trace = {
                'x': registrations,
                'type': 'histogram',
                'nbinsx': 30,
                'marker': {'color': '#00AA44'},
                'name': 'Monthly Registrations'
            }
            
            layout = {
                'title': 'Distribution of Monthly Vehicle Registrations',
                'xaxis': {'title': 'Number of Vehicles'},
                'yaxis': {'title': 'Frequency (months)'},
                'hovermode': 'x unified',
                'template': 'plotly_white',
                'height': 400,
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(240,240,240,0.5)'
            }
            
        elif chart_type == 'registration_by_category':
            if 'Standard_Category' not in data.columns or 'New_Registration' not in data.columns:
                return jsonify({'error': 'Category or registration data not found'}), 400
            
            # Get average registrations by category
            cat_data = data.groupby('Standard_Category')['New_Registration'].mean().sort_values(ascending=True)
            
            trace = {
                'x': cat_data.tolist(),
                'y': cat_data.index.tolist(),
                'type': 'bar',
                'orientation': 'h',
                'marker': {'color': '#00AA44'},
                'name': 'Avg Monthly Registrations'
            }
            
            layout = {
                'title': 'Average Monthly Registrations by Vehicle Category',
                'xaxis': {'title': 'Average Registrations'},
                'yaxis': {'title': 'Vehicle Category'},
                'hovermode': 'y unified',
                'template': 'plotly_white',
                'height': 450,
                'margin': {'l': 150},
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(240,240,240,0.5)'
            }
            
        elif chart_type == 'registration_trend':
            # Show registration trend over time
            if 'Year' not in data.columns or 'Month' not in data.columns or 'New_Registration' not in data.columns:
                return jsonify({'error': 'Time or registration data not found'}), 400
            
            # Sort by year and month
            sorted_data = data.sort_values(['Year', 'Month']).groupby(['Year', 'Month']).agg({
                'New_Registration': 'sum'
            }).reset_index()
            
            # Create time label
            sorted_data['time'] = sorted_data['Year'].astype(str) + '-' + sorted_data['Month'].astype(str).str.zfill(2)
            
            trace = {
                'x': sorted_data['time'].tolist(),
                'y': sorted_data['New_Registration'].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'line': {'color': '#00AA44', 'width': 3},
                'marker': {'size': 6, 'color': '#00AA44'},
                'name': 'Total Registrations',
                'fill': 'tozeroy',
                'fillcolor': 'rgba(0, 170, 68, 0.1)'
            }
            
            layout = {
                'title': 'Vehicle Registration Trend Over Time',
                'xaxis': {'title': 'Year-Month'},
                'yaxis': {'title': 'Number of Vehicles'},
                'hovermode': 'x unified',
                'template': 'plotly_white',
                'height': 400,
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(240,240,240,0.5)'
            }
        else:
            return jsonify({'error': 'Unknown chart type'}), 400
        
        return jsonify({
            'data': [trace],
            'layout': layout
        })
    
    except Exception as e:
        print(f"Chart error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/model-info')
def model_info():
    """Get model information."""
    try:
        model = load_model()
        data = load_data()
        
        info = {
            'algorithm': 'CatBoost (Bayesian Optimized + Log-Transform)',
            'status': 'Loaded' if model is not None else 'Not Found',
            'task': 'Vehicle Registration Forecasting',
            'r2_score': 0.9842,
            'mae': 160.10,
            'median_ape': 8.9,
            'features': 10,
            'training_samples': len(data) if data is not None else 0,
            'optimization': 'Bayesian Optimization (80 trials, Log-Transform + RMSE)',
            'hyperparameters': {
                'iterations': 1876,
                'learning_rate': 0.030562,
                'depth': 4,
                'l2_leaf_reg': 0.988819,
                'subsample': 0.733632,
                'colsample_bylevel': 0.666635,
                'random_strength': 5.755841,
                'min_data_in_leaf': 3
            }
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/explore')
def explore():
    """Data exploration page."""
    return render_template('explore.html')


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return render_template('error.html', error='Server error'), 500


if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists('models/catboost_model.bin'):
        print("⚠️  Warning: Model file not found at models/catboost_model.bin")
        print("Please train the model first using src/train.py")
    
    # Load model and data on startup
    load_model()
    load_data()
    
    print("="*60)
    print("🚀 Vehicle Registration Prediction - Flask App")
    print("="*60)
    print("✓ Model: CatBoost (Bayesian Optimized + Log-Transform)")
    print("✓ R² Score: 0.9842 | MAE: ±160.10 vehicles/month | MedAPE: 8.9%")
    print("✓ Features: 10 | Training Samples: 648")
    print("="*60)
    print("📍 Navigate to http://localhost:5000")
    print("📊 API: /api/explore - Data exploration")
    print("📈 API: /api/chart/registration_trend - Trend analysis")
    print("🎯 API: /predict - Make predictions")
    print("="*60)
    
    app.run(debug=True, port=5000)
