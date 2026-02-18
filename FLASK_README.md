# RetailPredict - Flask Web Application

A modern, AI-powered retail price prediction web application for Sri Lankan markets with SHAP explainability and interactive data visualization.

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model exists:**
   - Place your trained model at `models/catboost_model.bin`
   - Place your data at `data/master_dataset.csv`

3. **Run the Flask app:**
   ```bash
   python flask_app.py
   ```

4. **Open in browser:**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
RetailPredict-SL/
├── flask_app.py              # Main Flask application
├── app.py                    # Original Streamlit app (legacy)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── src/                      # Source code modules
│   ├── train.py             # Model training script
│   ├── preprocess.py        # Data preprocessing
│   └── explain.py           # SHAP explainability
│
├── templates/               # HTML templates
│   ├── base.html           # Base layout template
│   ├── index.html          # Home page
│   ├── predict.html        # Prediction page
│   ├── explore.html        # Data exploration
│   ├── about.html          # About page
│   └── error.html          # Error page
│
├── static/                 # Static files
│   ├── css/
│   │   └── style.css      # Modern styling
│   └── js/
│       └── main.js        # JavaScript utilities
│
├── data/                   # Data files
│   ├── master_dataset.csv
│   └── raw/
│
├── models/                 # Trained models
│   └── catboost_model.bin
│
├── notebooks/
│   └── exploration.ipynb
│
└── catboost_info/         # Training logs
```

## 🎯 Key Features

### 1. **Price Prediction**
- Input product details, region, market, and supply/demand levels
- Get real-time AI-powered price predictions
- View top factors influencing the prediction
- Based on CatBoost gradient boosting model

### 2. **SHAP Explainability**
- Understand why predictions are made
- Feature importance analysis
- Top 5 most influential factors displayed
- Interactive visualization

### 3. **Data Explorer**
- Price distribution visualization
- Price trends over time
- Product-wise price analysis
- Dataset statistics
- Data sample table

### 4. **Modern UI**
- Clean, responsive design
- Professional styling with custom CSS
- Interactive charts with Plotly
- Mobile-friendly interface
- Fast and lightweight (no Streamlit overhead)

## 🔧 API Endpoints

### Pages
- `GET /` - Home page
- `GET /predict` - Price prediction page
- `GET /explore` - Data exploration page
- `GET /about` - About page

### API Endpoints
- `POST /predict` - Make a prediction
  ```json
  {
    "Product": "Rice",
    "Region": "Western",
    "Market": "Colombo",
    "Quantity": 50,
    "Supply": 3.5,
    "Demand": 4.2
  }
  ```

- `GET /api/explore` - Get exploration data
- `GET /api/chart/<chart_type>` - Get chart data
  - `price_distribution`
  - `price_by_product`
  - `price_trend`
- `GET /api/model-info` - Get model information

## 📊 Model Performance

- **Algorithm:** CatBoost Regressor
- **R² Score:** 0.9573 (95.73%)
- **Mean Absolute Error:** ₨119.31
- **Root Mean Square Error:** ₨185.24
- **Training Samples:** 5000+
- **Features:** 15+ categorical and numeric features

## 🛠️ Development

### Running in Debug Mode

The app runs in debug mode by default (auto-reload on code changes):
```bash
python flask_app.py
```

### Production Deployment

For production, use a WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_app:app
```

## 📦 Dependencies

- **Backend:**
  - Flask - Web framework
  - CatBoost - Machine learning model
  - Pandas - Data manipulation
  - NumPy - Numerical computing
  - SHAP - Model explainability
  - scikit-learn - ML utilities

- **Frontend:**
  - HTML5
  - CSS3
  - JavaScript (Vanilla)
  - Plotly - Interactive visualization
  - Font Awesome - Icons

## 🔄 Migration from Streamlit

This Flask version replaces the original Streamlit application with:
- ✅ Better UI/UX with custom styling
- ✅ Faster performance (no Streamlit overhead)
- ✅ Full control over layout and interactivity
- ✅ Easier deployment
- ✅ Better for production use
- ✅ RESTful API for integrations

The original `app.py` (Streamlit) is preserved for reference.

## 🐛 Troubleshooting

### Port Already in Use
```bash
python flask_app.py  # Uses port 5000 by default
```

Change port in flask_app.py:
```python
app.run(debug=True, port=8000)
```

### Model Not Found
Ensure you have trained the model:
```bash
python src/train.py
```

### Slow Predictions
- SHAP calculations can be slow on large datasets
- Use a smaller sample for explainer if needed
- Consider caching predictions

## 📝 Configuration

Edit `flask_app.py` to customize:
- Model path: `'models/catboost_model.bin'`
- Data path: `'data/master_dataset.csv'`
- Port: Change `port=5000`
- Debug mode: Change `debug=True`

## 🚀 Future Enhancements

- [ ] User authentication
- [ ] Batch prediction API
- [ ] Download prediction results
- [ ] Historical prediction tracking
- [ ] Advanced analytics dashboard
- [ ] Model retraining pipeline
- [ ] Docker containerization
- [ ] Database integration

## 📄 License

This project uses the data and model from the RetailPredict project.

## 👨‍💻 Author

Built with Python, Flask, CatBoost, and SHAP for Sri Lankan retail market analysis.
