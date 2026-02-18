"""
RetailPredict - Streamlit Web Application

Interactive web app for Sri Lankan retail price prediction.
Features:
- Real-time price predictions
- SHAP-based explainability
- Data visualization and exploration
- Price trend analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostRegressor
    import shap
except ImportError:
    st.error("Required packages not found. Install with: pip install catboost shap")
    st.stop()


@st.cache_resource
def load_model():
    """Load the trained CatBoost model."""
    model_path = 'models/catboost_model.bin'
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model


@st.cache_data
def load_data():
    """Load the dataset."""
    df = pd.read_csv('data/master_dataset.csv')
    return df


@st.cache_resource
def compute_shap_explainer(_model, _X):
    """Compute SHAP explainer."""
    explainer = shap.TreeExplainer(_model)
    return explainer


def apply_modern_theme():
    """Apply custom modern light theme styling."""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #0066CC;
        --secondary-color: #00D9FF;
        --accent-color: #FF6B6B;
        --success-color: #51CF66;
        --dark-bg: #FFFFFF;
        --light-text: #1a1a1a;
    }
    
    /* Overall app styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .stApp header {
        background-color: white;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: white !important;
    }
    
    /* Title styling */
    h1 {
        color: #0066CC !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        border-bottom: 3px solid #00D9FF;
        padding-bottom: 0.5rem !important;
    }
    
    h2 {
        color: #0066CC !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }
    
    h3 {
        color: #00557A !important;
        font-weight: 600 !important;
    }
    
    /* Text styling */
    body {
        color: #1a1a1a;
        background-color: #f8f9fa;
    }
    
    /* Metric cards styling */
    .css-1wrcr25 {
        background-color: white !important;
        border-left: 4px solid #0066CC !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #0066CC !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(0,102,204,0.2) !important;
    }
    
    .stButton > button:hover {
        background-color: #0052A3 !important;
        box-shadow: 0 4px 12px rgba(0,102,204,0.3) !important;
    }
    
    /* Selectbox and input styling */
    .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
        border-radius: 8px !important;
    }
    
    .stSelectbox > div > div {
        border-color: #e9ecef !important;
        border-radius: 8px !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f7ff !important;
        border-radius: 8px !important;
        color: #0066CC !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e3f1ff !important;
    }
    
    /* Divider styling */
    hr {
        border: 0;
        border-top: 2px solid #e9ecef;
        margin: 2rem 0 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        color: #666 !important;
        border-bottom-color: #e9ecef !important;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #0066CC !important;
        border-bottom-color: #0066CC !important;
    }
    
    /* Data table styling */
    .stDataframe {
        border-radius: 8px !important;
        border: 1px solid #e9ecef !important;
    }
    
    /* Subheader styling */
    .stSubheader {
        color: #0066CC !important;
        border-bottom: 2px solid #f0f7ff;
        padding-bottom: 0.5rem !important;
    }
    
    /* Markdown links */
    a {
        color: #0066CC !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #0052A3 !important;
        text-decoration: underline !important;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background-color: #f0f7ff !important;
        border-left: 4px solid #0066CC !important;
        border-radius: 8px !important;
    }
    
    /* Warning box */
    .stWarning {
        background-color: #fff8e1 !important;
        border-left: 4px solid #FFB81C !important;
        border-radius: 8px !important;
    }
    
    /* Error box */
    .stError {
        background-color: #ffe3e3 !important;
        border-left: 4px solid #FF6B6B !important;
        border-radius: 8px !important;
    }
    
    /* Radio button styling */
    .stRadio > label {
        color: #1a1a1a !important;
    }
    
    /* Column cards effect */
    .card-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #0066CC, #00D9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="RetailPredict - Price Forecasting",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply modern theme
    apply_modern_theme()
    
    # Title and intro with custom styling
    st.markdown("<h1>🛒 RetailPredict - Sri Lankan Retail Price Forecasting</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; background: linear-gradient(135deg, rgba(0,102,204,0.05), rgba(0,217,255,0.05)); 
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
            <h3 style="margin-top: 0; color: #0066CC;">✨ AI-Powered Price Prediction & Analysis Tool</h3>
            <p style="color: #555; font-size: 0.95rem; line-height: 1.6;">
            Intelligent forecasting for Sri Lankan retail markets using advanced machine learning
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar navigation with modern styling
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #0066CC; margin-bottom: 2rem;'>🔧 Navigation</h2>", unsafe_allow_html=True)
        
        # Create custom page buttons
        pages = {
            "📈 Price Prediction": "prediction",
            "🔍 SHAP Explanations": "shap",
            "📊 Data Explorer": "explorer",
            "📚 About": "about"
        }
        
        page = st.radio(
            "Select a section:",
            list(pages.keys()),
            label_visibility="collapsed"
        )
        
        # Sidebar info box
        st.markdown("""
        <div style="background-color: #f0f7ff; border-left: 4px solid #0066CC; padding: 1rem; 
        border-radius: 8px; margin-top: 2rem;">
            <h4 style="color: #0066CC; margin-top: 0;">📌 Quick Info</h4>
            <p style="font-size: 0.85rem; color: #555; margin: 0;">
                <strong>Model:</strong> CatBoost<br>
                <strong>R² Score:</strong> 0.9573<br>
                <strong>MAE:</strong> ₨119.31<br>
                <strong>Products:</strong> 116
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load resources
    model = load_model()
    df = load_data()
    X = df.drop(['product_name', 'W4_Jan_2026'], axis=1)
    X = X.fillna(X.mean(numeric_only=True))
    
    # PAGE 1: PRICE PREDICTION
    if page == "📈 Price Prediction":
        st.markdown("<h2>📈 Real-Time Price Prediction</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.05rem; color: #555; margin-bottom: 2rem;'>Predict the retail price for next week based on current market data</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown("<div class='card-container'><h4 style='color: #0066CC; margin-top: 0;'>📋 Product Selection</h4>", unsafe_allow_html=True)
            selected_product = st.selectbox(
                "Select a product:",
                df['product_name'].unique(),
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            product_idx = df[df['product_name'] == selected_product].index[0]
        
        with col2:
            st.markdown("<div class='card-container'><h4 style='color: #0066CC; margin-top: 0;'>📊 Current Prices</h4>", unsafe_allow_html=True)
            product_data = df.iloc[product_idx]
            st.metric("Price 2025-01-01", f"₨{product_data['Price_2025_01_01']:.0f}")
            st.metric("Week 1 Jan 2026", f"₨{product_data['W1_Jan_2026']:.0f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='card-container'><h4 style='color: #0066CC; margin-top: 0;'>📈 Recent Prices</h4>", unsafe_allow_html=True)
            st.metric("Week 2 Jan 2026", f"₨{product_data['W2_Jan_2026']:.0f}")
            st.metric("Week 3 Jan 2026", f"₨{product_data['W3_Jan_2026']:.0f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Prediction
        st.divider()
        st.markdown("<h3>🎯 Model Prediction Results</h3>", unsafe_allow_html=True)
        
        features = X.iloc[product_idx:product_idx+1]
        prediction = model.predict(features)[0]
        actual = product_data['W4_Jan_2026']
        
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0066CC, #0052A3); padding: 2rem; border-radius: 12px; 
            text-align: center; color: white; box-shadow: 0 4px 12px rgba(0,102,204,0.2);">
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">🔮 Predicted Price</p>
                <h2 style="margin: 0.5rem 0; color: white; font-size: 2rem;">₨{prediction:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #51CF66, #37B24D); padding: 2rem; border-radius: 12px; 
            text-align: center; color: white; box-shadow: 0 4px 12px rgba(81,207,102,0.2);">
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">✓ Actual Price</p>
                <h2 style="margin: 0.5rem 0; color: white; font-size: 2rem;">₨{actual:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            error = abs(prediction - actual)
            error_pct = (error / actual) * 100 if actual != 0 else 0
            color = "#FF6B6B" if error_pct > 10 else "#6BCB77" if error_pct < 5 else "#FFB81C"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}, {color}dd); padding: 2rem; border-radius: 12px; 
            text-align: center; color: white; box-shadow: 0 4px 12px rgba(255, 107, 107, 0.2);">
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Prediction Error</p>
                <h2 style="margin: 0.5rem 0; color: white; font-size: 2rem;">±{error_pct:.1f}%</h2>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">₨{error:.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Price trend
        st.divider()
        st.markdown("<h3>📈 Price Trend Analysis</h3>", unsafe_allow_html=True)
        
        dates = ['2025-01-01', 'W1 Jan', 'W2 Jan', 'W3 Jan', 'W4 Jan (Predicted)']
        prices = [
            product_data['Price_2025_01_01'],
            product_data['W1_Jan_2026'],
            product_data['W2_Jan_2026'],
            product_data['W3_Jan_2026'],
            prediction
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='lines+markers',
            name='Price',
            line=dict(color='#0066CC', width=4),
            marker=dict(size=12, color='#0066CC', 
                       line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(0, 102, 204, 0.1)',
            hovertemplate='<b>%{x}</b><br>Price: ₨%{y:.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"<b>{selected_product}</b> - Price Trend",
            xaxis_title="Timeline",
            yaxis_title="Price (₨)",
            hovermode='x unified',
            template='plotly_white',
            plot_bgcolor='rgba(240, 247, 255, 0.3)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", color="#1a1a1a"),
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Category info
        st.divider()
        st.markdown("<h3>📦 Product Information Card</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        category_colors = {
            'Vegetables': '#FF6B6B',
            'Fish': '#4ECDC4',
            'Meat/Eggs': '#D97706',
            'Dairy': '#F087DC',
            'Spices/Condiments': '#8B5CF6',
            'Grains/Pulses': '#6366F1',
            'Leafy Greens': '#10B981',
            'Fruits': '#F59E0B',
            'Coconut Products': '#8B4513',
            'Sugar': '#EC4899',
            'Bakery': '#D2691E'
        }
        
        cat_color = category_colors.get(product_data['category'], '#0066CC')
        
        with col1:
            st.markdown(f"""
            <div style="background-color: rgba({cat_color.lstrip('#')}, 0.1); border-left: 4px solid {cat_color}; 
            padding: 1rem; border-radius: 8px;">
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Category</p>
                <h4 style="margin: 0.5rem 0; color: {cat_color};">{product_data['category']}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            perishable = "🍎 Yes" if product_data['perishability'] == 'Perishable' else "📦 No"
            st.markdown(f"""
            <div style="background-color: #f0f7ff; border-left: 4px solid #0066CC; 
            padding: 1rem; border-radius: 8px;">
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Perishable</p>
                <h4 style="margin: 0.5rem 0; color: #0066CC;">{perishable}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color: #f0f7ff; border-left: 4px solid #0066CC; 
            padding: 1rem; border-radius: 8px;">
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Price Level</p>
                <h4 style="margin: 0.5rem 0; color: #0066CC;">{product_data['price_category']}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background-color: #f0f7ff; border-left: 4px solid #0066CC; 
            padding: 1rem; border-radius: 8px;">
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Avg Price</p>
                <h4 style="margin: 0.5rem 0; color: #0066CC;">₨{product_data['avg_price']:.0f}</h4>
            </div>
            """, unsafe_allow_html=True)
    
    # PAGE 2: SHAP EXPLANATIONS
    elif page == "🔍 SHAP Explanations":
        st.markdown("<h2>🔍 Model Explainability - SHAP Analysis</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size: 1rem; color: #555; margin-bottom: 2rem;'>
        <strong>SHAP (SHapley Additive exPlanations)</strong> explains which features influence price predictions. 
        Understanding model decisions builds trust and transparency.
        </p>
        """, unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("<h3>⭐ Feature Importance Rankings</h3>", unsafe_allow_html=True)
        
        explainer = compute_shap_explainer(model, X)
        shap_values = explainer.shap_values(X)
        
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Impact': np.abs(shap_values).mean(axis=0)
        }).sort_values('Impact', ascending=False)
        
        # Create a more attractive bar chart
        fig = px.bar(
            importance.head(10),
            x='Impact',
            y='Feature',
            orientation='h',
            title='<b>Top 10 Most Important Features</b><br><sub>Average Impact on Price Predictions</sub>',
            labels={'Impact': 'Mean |SHAP value|', 'Feature': ''},
            color='Impact',
            color_continuous_scale=['#FFE5E5', '#FF6B6B', '#0066CC', '#0052A3'],
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(240, 247, 255, 0.5)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", color="#1a1a1a"),
            height=400,
            showlegend=False,
            margin=dict(l=150, r=50, t=100, b=50)
        )
        
        fig.update_traces(
            marker_line_color='white',
            marker_line_width=2,
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.2f}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top features explanation
        st.divider()
        st.markdown("<h3>📊 Understanding Feature Impact</h3>", unsafe_allow_html=True)
        
        top_5 = importance.head(5)
        
        col1, col2 = st.columns(2)
        
        for idx, (i, row) in enumerate(top_5.iterrows()):
            col = col1 if idx % 2 == 0 else col2
            
            icon = "📅" if ('W' in row['Feature'] and 'Jan' in row['Feature']) else \
                   "📊" if row['Feature'] == 'avg_price' else \
                   "🏷️" if row['Feature'] == 'price_category' else \
                   "📈" if row['Feature'] == 'Price_2025_01_01' else "💡"
            
            # Determine description
            if 'W' in row['Feature'] and 'Jan' in row['Feature']:
                desc = "📅 <strong>Recent weekly prices</strong> - Most directly influence next week's price"
            elif row['Feature'] == 'avg_price':
                desc = "📊 <strong>Average price</strong> - Captures overall price level"
            elif row['Feature'] == 'price_category':
                desc = "🏷️ <strong>Price category</strong> - Whether product is low/medium/high priced"
            elif row['Feature'] == 'Price_2025_01_01':
                desc = "📈 <strong>Annual comparison</strong> - Year-over-year trends"
            else:
                desc = "💡 <strong>Feature influence</strong> - Impacts price predictions"
            
            with col.container():
                st.markdown(f"""
                <div style="background: white; border-left: 4px solid #0066CC; padding: 1.2rem; 
                border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                        <h4 style="margin: 0; color: #0066CC; flex: 1;">{row['Feature']}</h4>
                        <span style="background: #0066CC; color: white; padding: 0.25rem 0.75rem; 
                        border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                        Impact: {row['Impact']:.2f}
                        </span>
                    </div>
                    {desc}
                </div>
                """, unsafe_allow_html=True)
        
        # Single prediction explanation
        st.divider()
        st.markdown("<h3>🎯 Single Prediction Explanation</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #555;'>Select a product to see detailed SHAP values for that specific prediction</p>", unsafe_allow_html=True)
        
        selected_product = st.selectbox(
            "Select product for detailed explanation:",
            df['product_name'].unique(),
            key='explain_product'
        )
        
        prod_idx = df[df['product_name'] == selected_product].index[0]
        
        # Get SHAP contribution
        pred = model.predict(X.iloc[[prod_idx]])[0]
        baseline = explainer.expected_value
        shap_vals = shap_values[prod_idx]
        features_vals = X.iloc[prod_idx]
        
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0066CC, #0052A3); padding: 1.5rem; border-radius: 12px; 
            text-align: center; color: white; box-shadow: 0 4px 12px rgba(0,102,204,0.2);">
                <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">Predicted Price</p>
                <h3 style="margin: 0.5rem 0; color: white;">₨{pred:.0f}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #6BCB77, #51CF66); padding: 1.5rem; border-radius: 12px; 
            text-align: center; color: white; box-shadow: 0 4px 12px rgba(81, 207, 102, 0.2);">
                <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">Model Baseline</p>
                <h3 style="margin: 0.5rem 0; color: white;">₨{baseline:.0f}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            deviation = pred - baseline
            deviation_symbol = "⬆️" if deviation > 0 else "⬇️"
            color = "#FF6B6B" if deviation < 0 else "#51CF66"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}, {color}dd); padding: 1.5rem; border-radius: 12px; 
            text-align: center; color: white; box-shadow: 0 4px 12px rgba(255, 107, 107, 0.2);">
                <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">Prediction Shift</p>
                <h3 style="margin: 0.5rem 0; color: white;">{deviation_symbol} ₨{abs(deviation):.0f}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Feature contributions
        contrib = pd.DataFrame({
            'Feature': X.columns,
            'Value': features_vals.values,
            'SHAP': shap_vals
        }).sort_values('SHAP', key=abs, ascending=False)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div style="background: white; border-left: 4px solid #51CF66; padding: 1.5rem; 
            border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <h4 style="color: #51CF66; margin-top: 0;">⬆️ Pushing Price UP</h4>
            """, unsafe_allow_html=True)
            
            positive = contrib[contrib['SHAP'] > 0].head(3)
            if len(positive) > 0:
                for _, row in positive.iterrows():
                    try:
                        value_float = float(row['Value'])
                        st.markdown(f"• **{row['Feature']}** → +₨{row['SHAP']:.0f}", help=f"Value: {value_float:.2f}")
                    except (ValueError, TypeError):
                        st.markdown(f"• **{row['Feature']}** → +₨{row['SHAP']:.0f}")
            else:
                st.markdown("No features pushing price UP")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: white; border-left: 4px solid #FF6B6B; padding: 1.5rem; 
            border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <h4 style="color: #FF6B6B; margin-top: 0;">⬇️ Pushing Price DOWN</h4>
            """, unsafe_allow_html=True)
            
            negative = contrib[contrib['SHAP'] < 0].head(3)
            if len(negative) > 0:
                for _, row in negative.iterrows():
                    try:
                        value_float = float(row['Value'])
                        st.markdown(f"• **{row['Feature']}** → ₨{row['SHAP']:.0f}", help=f"Value: {value_float:.2f}")
                    except (ValueError, TypeError):
                        st.markdown(f"• **{row['Feature']}** → ₨{row['SHAP']:.0f}")
            else:
                st.markdown("No features pushing price DOWN")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # PAGE 3: DATA EXPLORER
    elif page == "📊 Data Explorer":
        st.markdown("<h2>📊 Dataset Exploration & Analysis</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1rem; color: #555; margin-bottom: 2rem;'>Explore the dataset structure, statistics, and price distributions</p>", unsafe_allow_html=True)
        
        # Statistics
        st.markdown("<h3>📈 Dataset Overview</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        stats = [
            ("📦", "Total Products", len(df), "#0066CC"),
            ("📂", "Categories", df['category'].nunique(), "#51CF66"),
            ("🔢", "Features", X.shape[1], "#FFB81C"),
            ("📅", "Date Range", "2025-01 to 2026-W4", "#FF6B6B")
        ]
        
        list_cols = [col1, col2, col3, col4]
        
        for idx, (icon, label, value, color) in enumerate(stats):
            with list_cols[idx]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}22, {color}11); border-left: 4px solid {color}; 
                padding: 1.5rem; border-radius: 8px; text-align: center;">
                    <p style="margin: 0; font-size: 1.2rem;">{icon}</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #666;">{label}</p>
                    <h3 style="margin: 0.5rem 0; color: {color};">{value}</h3>
                </div>
                """, unsafe_allow_html=True)
        
        # Category distribution
        st.divider()
        st.markdown("<h3>📂 Product Distribution by Category</h3>", unsafe_allow_html=True)
        
        category_counts = df['category'].value_counts()
        fig = px.pie(
            names=category_counts.index,
            values=category_counts.values,
            title="<b>Products by Category</b>",
            hole=0.35,
            color_discrete_sequence=['#0066CC', '#00D9FF', '#FF6B6B', '#51CF66', '#FFB81C', '#8B5CF6', '#EC4899', '#10B981', '#F59E0B', '#D97706', '#4ECDC4']
        )
        fig.update_layout(
            showlegend=True,
            height=450,
            font=dict(family="Arial, sans-serif", color="#1a1a1a"),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics
        st.divider()
        st.markdown("<h3>💰 Price Statistics & Distributions</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            fig = px.box(
                y=df['W4_Jan_2026'],
                title="<b>Price Box Plot</b><br><sub>Week 4 January 2026</sub>",
                labels={'y': 'Price (₨)'},
                points='outliers'
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                template='plotly_white',
                plot_bgcolor='rgba(240, 247, 255, 0.3)',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                x=df['W4_Jan_2026'],
                title="<b>Price Frequency Distribution</b><br><sub>Histogram with 25 bins</sub>",
                nbins=25,
                labels={'x': 'Price (₨)', 'count': 'Count'},
                color_discrete_sequence=['#0066CC']
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                template='plotly_white',
                plot_bgcolor='rgba(240, 247, 255, 0.3)',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data statistics
        st.markdown("<h3>📊 Price Statistics Summary</h3>", unsafe_allow_html=True)
        
        stats_data = df['W4_Jan_2026'].describe()
        
        col1, col2, col3, col4, col5 = st.columns(5, gap="small")
        
        with col1:
            st.markdown(f"""
            <div style="background: white; border-left: 4px solid #0066CC; padding: 1rem; 
            border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="margin: 0; font-size: 0.8rem; color: #666;">Count</p>
                <h4 style="margin: 0.5rem 0; color: #0066CC;">{int(stats_data['count'])}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: white; border-left: 4px solid #51CF66; padding: 1rem; 
            border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="margin: 0; font-size: 0.8rem; color: #666;">Mean</p>
                <h4 style="margin: 0.5rem 0; color: #51CF66;">₨{stats_data['mean']:.0f}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: white; border-left: 4px solid #FFB81C; padding: 1rem; 
            border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="margin: 0; font-size: 0.8rem; color: #666;">Median</p>
                <h4 style="margin: 0.5rem 0; color: #FFB81C;">₨{stats_data['50%']:.0f}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: white; border-left: 4px solid #0066CC; padding: 1rem; 
            border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="margin: 0; font-size: 0.8rem; color: #666;">Min</p>
                <h4 style="margin: 0.5rem 0; color: #0066CC;">₨{stats_data['min']:.0f}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div style="background: white; border-left: 4px solid #FF6B6B; padding: 1rem; 
            border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="margin: 0; font-size: 0.8rem; color: #666;">Max</p>
                <h4 style="margin: 0.5rem 0; color: #FF6B6B;">₨{stats_data['max']:.0f}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Data table
        st.divider()
        st.markdown("<h3>📋 Complete Dataset View</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #555; font-size: 0.95rem;'>Select columns to display and explore the data</p>", unsafe_allow_html=True)
        
        show_cols = st.multiselect(
            "Select columns to display:",
            df.columns.tolist(),
            default=['product_name', 'category', 'Price_2025_01_01', 'W4_Jan_2026'],
            label_visibility="collapsed"
        )
        
        if show_cols:
            st.dataframe(
                df[show_cols].style.format(
                    {col: "₨{:.0f}" for col in show_cols if col not in ['product_name', 'category', 'price_category', 'perishability']}
                ),
                use_container_width=True,
                height=400
            )
    
    # PAGE 4: ABOUT
    elif page == "📚 About":
        st.markdown("<h2>📚 About RetailPredict</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0,102,204,0.05), rgba(0,217,255,0.05)); 
        padding: 2rem; border-radius: 12px; border-left: 4px solid #0066CC; margin-bottom: 2rem;">
            <h3 style="color: #0066CC; margin-top: 0;">🎯 About This Project</h3>
            <p style="color: #555; font-size: 1rem; line-height: 1.7;">
            RetailPredict is an AI-powered price forecasting system designed for Sri Lankan retail markets. 
            Built with transparent, explainable AI, it helps consumers, businesses, and policymakers understand 
            and predict grocery price movements.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Features
        st.markdown("<h3>🌟 Key Features</h3>", unsafe_allow_html=True)
        
        features_list = [
            ("🔮", "Real-Time Predictions", "Get accurate price forecasts for any product instantly"),
            ("🔍", "SHAP Explainability", "Understand exactly why the model makes its predictions"),
            ("📊", "Data Visualization", "Explore price trends and market dynamics visually"),
            ("🏪", "Market Analysis", "Track price patterns across categories and time periods")
        ]
        
        col1, col2 = st.columns(2, gap="large")
        
        for idx, (icon, title, desc) in enumerate(features_list):
            col = col1 if idx % 2 == 0 else col2
            with col.container():
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                border-left: 4px solid #0066CC; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem;">
                    <div style="display: flex; align-items: flex-start;">
                        <span style="font-size: 2rem; margin-right: 1rem;">{icon}</span>
                        <div>
                            <h4 style="margin: 0 0 0.5rem 0; color: #0066CC;">{title}</h4>
                            <p style="margin: 0; color: #555; font-size: 0.95rem;">{desc}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Model Technical Details
        st.markdown("<h3>🤖 Technical Architecture</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div style="background: white; border-left: 4px solid #0066CC; padding: 1.5rem; 
            border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <h4 style="color: #0066CC; margin-top: 0;">📊 Machine Learning Model</h4>
                <ul style="color: #555; line-height: 2;">
                    <li><strong>Algorithm:</strong> CatBoost (Categorical Boosting)</li>
                    <li><strong>Features:</strong> 12 engineered features</li>
                    <li><strong>Target:</strong> W4_Jan_2026 (next week price)</li>
                    <li><strong>Learning Rate:</strong> 0.05</li>
                    <li><strong>Iterations:</strong> 1000</li>
                    <li><strong>Tree Depth:</strong> 6</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: white; border-left: 4px solid #51CF66; padding: 1.5rem; 
            border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <h4 style="color: #51CF66; margin-top: 0;">📈 Model Performance</h4>
                <ul style="color: #555; line-height: 2;">
                    <li><strong>R² Score:</strong> <span style="color: #51CF66; font-weight: 700;">0.9573</span></li>
                    <li><strong>Mean Absolute Error:</strong> <span style="color: #51CF66; font-weight: 700;">₨119.31</span></li>
                    <li><strong>Root Mean Squared Error:</strong> ₨165.42</li>
                    <li><strong>Train/Test Split:</strong> 80%/20%</li>
                    <li><strong>Dataset Size:</strong> 116 products</li>
                    <li><strong>Explainability:</strong> SHAP TreeExplainer</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Data Information
        st.markdown("<h3>📊 Dataset Information</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div style="background: #f0f7ff; border-left: 4px solid #0066CC; padding: 1.5rem; 
            border-radius: 8px;">
                <h4 style="color: #0066CC; margin-top: 0;">📌 Data Source</h4>
                <ul style="color: #555; line-height: 2; padding-left: 1.5rem;">
                    <li>Department of Census and Statistics, Sri Lanka</li>
                    <li>122 food products (116 after cleaning)</li>
                    <li>11 major product categories</li>
                    <li>Weekly price reports</li>
                    <li>Date range: 2025-01-01 to 2026-W4</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #f0f7ff; border-left: 4px solid #0066CC; padding: 1.5rem; 
            border-radius: 8px;">
                <h4 style="color: #0066CC; margin-top: 0;">🏪 Product Categories</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; color: #555;">
                    <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 4px; border-left: 3px solid #FF6B6B;">🥬 Vegetables</span>
                    <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 4px; border-left: 3px solid #4ECDC4;">🐟 Fish</span>
                    <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 4px; border-left: 3px solid #D97706;">🥩 Meat/Eggs</span>
                    <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 4px; border-left: 3px solid #F087DC;">🥛 Dairy</span>
                    <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 4px; border-left: 3px solid #8B5CF6;">🌶️ Spices</span>
                    <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 4px; border-left: 3px solid #6366F1;">🌾 Grains</span>
                    <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 4px; border-left: 3px solid #10B981;">🥗 Leafy Greens</span>
                    <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 4px; border-left: 3px solid #F59E0B;">🍎 Fruits</span>
                    <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 4px; border-left: 3px solid #8B4513;">🥥 Coconut</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Use Cases
        st.markdown("<h3>💡 Use Cases & Applications</h3>", unsafe_allow_html=True)
        
        use_cases = [
            ("👨‍👩‍👧‍👦", "Consumer Protection", "Help households budget and make informed purchasing decisions"),
            ("🏢", "Business Planning", "Forecast procurement costs and optimize supply chain"),
            ("📊", "Market Analysis", "Understand price dynamics and market trends"),
            ("🏛️", "Policy Making", "Inform economic and social policies based on data")
        ]
        
        col1, col2 = st.columns(2, gap="large")
        
        for idx, (icon, title, desc) in enumerate(use_cases):
            col = col1 if idx < 2 else col2
            with col.container():
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; 
                border-left: 4px solid #0066CC; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem;">
                    <h4 style="color: #0066CC; margin: 0 0 0.5rem 0;">{icon} {title}</h4>
                    <p style="margin: 0; color: #555; font-size: 0.95rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Limitations
        st.markdown("<h3>⚠️ Model Limitations</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #fff8e1; border-left: 4px solid #FFB81C; padding: 1.5rem; 
        border-radius: 8px;">
            <ul style="color: #555; line-height: 1.9;">
                <li>Dataset size is limited to 116 products after cleaning</li>
                <li>Historical data may not capture extreme market shocks or crises</li>
                <li>Predictions assume relatively stable market conditions</li>
                <li>External factors (policy changes, supply shocks) not captured in historical data</li>
                <li>Model is trained on weekly data and may not capture daily fluctuations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #999; padding: 2rem 0; border-top: 2px solid #e9ecef;">
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">
                <strong>RetailPredict v1.0</strong> | February 2026<br>
                Built with ❤️ for transparent, explainable AI<br>
                Department of Computer Science
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

