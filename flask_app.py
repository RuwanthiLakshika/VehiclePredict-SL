"""
RetailPredict - Flask Web Application

Interactive web app for Sri Lankan retail price prediction.
Features:
- Real-time price predictions
- SHAP-based explainability
- Data visualization and exploration
- Price trend analysis
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
            X = data.drop(columns=['Price'], errors='ignore')
            # Select numeric columns only for SHAP
            X_numeric = X.select_dtypes(include=[np.number])
            EXPLAINER = shap.TreeExplainer(model)
    return EXPLAINER


def prepare_features(data_dict):
    """Prepare features for prediction."""
    try:
        # Convert to DataFrame
        df_input = pd.DataFrame([data_dict])
        
        # Load training data for scaling/encoding reference
        data = load_data()
        
        # For now, just return the input - in real scenario align with training features
        # You'll need to extract the actual feature columns used during training
        return df_input
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
        stats = {
            'total_products': len(data['Product'].unique()) if 'Product' in data.columns else 0,
            'total_records': len(data),
            'avg_price': f"{data['Price'].mean():.2f}" if 'Price' in data.columns else "N/A",
            'price_range': f"{data['Price'].min():.2f} - {data['Price'].max():.2f}" if 'Price' in data.columns else "N/A"
        }
    
    return render_template('index.html', stats=stats, model_loaded=model is not None)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page and API endpoint."""
    if request.method == 'GET':
        # Load data for dropdown options
        data = load_data()
        
        products = []
        categories = []
        perishability = []
        
        if data is not None:
            if 'product_name' in data.columns:
                products = sorted(data['product_name'].unique().tolist())
            if 'category' in data.columns:
                categories = sorted(data['category'].unique().tolist())
            if 'perishability' in data.columns:
                perishability = sorted([x for x in data['perishability'].unique().tolist() if pd.notna(x)])
        
        return render_template('predict.html', 
                             products=products,
                             categories=categories,
                             perishability=perishability)
    
    elif request.method == 'POST':
        """Make prediction."""
        try:
            data_dict = request.get_json()
            model = load_model()
            
            if model is None:
                return jsonify({'error': 'Model not loaded. Please train the model first.'}), 400
            
            # Prepare features
            X = prepare_features(data_dict)
            if X is None:
                return jsonify({'error': 'Invalid input data'}), 400
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Get feature importance for this prediction
            explainer = get_explainer()
            importance = {}
            
            if explainer is not None:
                try:
                    shap_values = explainer.shap_values(X)
                    feature_names = X.columns.tolist()
                    
                    # Get top features
                    for i, feat in enumerate(feature_names):
                        if isinstance(shap_values, list):
                            importance[feat] = float(np.abs(shap_values[0][i]))
                        else:
                            importance[feat] = float(np.abs(shap_values[i]))
                    
                    # Sort and get top 5
                    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
                except Exception as e:
                    print(f"SHAP error: {e}")
                    importance = {}
            
            return jsonify({
                'success': True,
                'prediction': round(float(prediction), 2),
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
        
        if chart_type == 'price_distribution':
            # Use avg_price column if available
            price_col = 'avg_price' if 'avg_price' in data.columns else 'Price_2025_01_01'
            if price_col not in data.columns:
                return jsonify({'error': f'{price_col} column not found'}), 400
            
            prices = data[price_col].dropna().tolist()
            
            trace = {
                'x': prices,
                'type': 'histogram',
                'nbinsx': 50,
                'marker': {'color': '#0066CC'},
                'name': 'Price'
            }
            
            layout = {
                'title': 'Price Distribution',
                'xaxis': {'title': 'Price (₨)'},
                'yaxis': {'title': 'Frequency'},
                'hovermode': 'x unified',
                'template': 'plotly_white',
                'height': 400,
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(240,240,240,0.5)'
            }
            
        elif chart_type == 'price_by_product':
            if 'product_name' not in data.columns:
                return jsonify({'error': 'Product column not found'}), 400
            
            price_col = 'avg_price' if 'avg_price' in data.columns else 'Price_2025_01_01'
            
            # Get top 15 products by price
            top_data = data.nlargest(15, price_col)[['product_name', price_col]].sort_values(price_col)
            
            trace = {
                'x': top_data[price_col].tolist(),
                'y': top_data['product_name'].tolist(),
                'type': 'bar',
                'orientation': 'h',
                'marker': {'color': '#0066CC'},
                'name': 'Average Price'
            }
            
            layout = {
                'title': 'Top 15 Products by Average Price',
                'xaxis': {'title': 'Price (₨)'},
                'yaxis': {'title': 'Product'},
                'hovermode': 'y unified',
                'template': 'plotly_white',
                'height': 450,
                'margin': {'l': 200},
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(240,240,240,0.5)'
            }
            
        elif chart_type == 'price_trend':
            # Get weekly price columns
            weekly_cols = sorted([col for col in data.columns if col.startswith('W')])
            if not weekly_cols:
                return jsonify({'error': 'No weekly price data found'}), 400
            
            # Calculate average across products for trend
            trend_vals = data[weekly_cols].mean().tolist()
            
            trace = {
                'x': list(range(len(trend_vals))),
                'y': trend_vals,
                'type': 'scatter',
                'mode': 'lines+markers',
                'line': {'color': '#0066CC', 'width': 3},
                'marker': {'size': 8, 'color': '#0066CC'},
                'name': 'Average Price',
                'fill': 'tozeroy',
                'fillcolor': 'rgba(0, 102, 204, 0.1)'
            }
            
            layout = {
                'title': 'Price Trend Over Weeks',
                'xaxis': {'title': 'Week'},
                'yaxis': {'title': 'Price (₨)'},
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
            'algorithm': 'CatBoost (Gradient Boosting)',
            'status': 'Loaded' if model is not None else 'Not Found',
            'features': len(data.columns) - 1 if data is not None else 0,
            'training_samples': len(data) if data is not None else 0
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
    
    print("🚀 Starting RetailPredict Flask App...")
    print("📍 Navigate to http://localhost:5000")
    
    app.run(debug=True, port=5000)
