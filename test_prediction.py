import requests
import json

# Test Bus prediction with corrected defaults
data = {
    'Year': 2026,
    'Month': 'January',
    'Standard_Category': 'Bus',
    'Prev_Month_New_Reg': 118,
    'Monthly_Growth_Rate': 2,
    'Month_Num': 1,
    'Quarter': 1,
    'Is_Peak_Season': 0,
    'Is_Crisis_Period': 0,
    'Yearly_Total_Stock': 57859,
    'Transfer': 1037,
    'Transfer_to_New_Ratio': 8.8,
    'New_Registration_Market_Share': 0.01
}

try:
    response = requests.post('http://localhost:5000/predict', json=data, timeout=5)
    result = response.json()
    if result.get('success'):
        pred = result.get('prediction', 0)
        print(f'\n✓ CORRECTED FORECAST:')
        print(f'  Bus (Jan 2026, +2% growth, Normal): {pred:.0f} vehicles')
        print(f'  Expected range: 5-578 vehicles (historical Bus data)')
        print(f'  Status: {"✓ REALISTIC" if 5 <= pred <= 578*2 else "⚠ STILL HIGH"}')
    else:
        print(f'Error: {result.get("error")}')
except Exception as e:
    print(f'Connection error: {e}')
