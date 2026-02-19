import requests

test_cases = [
    {
        'name': 'Motor Cycle (Jan 2026, +5% growth)',
        'data': {
            'Year': 2026, 'Month': 'January', 'Standard_Category': 'Motor Cycle',
            'Prev_Month_New_Reg': 16723, 'Monthly_Growth_Rate': 5,
            'Month_Num': 1, 'Quarter': 1, 'Is_Peak_Season': 0, 'Is_Crisis_Period': 0,
            'Yearly_Total_Stock': 1569283, 'Transfer': 15254,
            'Transfer_to_New_Ratio': 0.91, 'New_Registration_Market_Share': 0.60
        },
        'expected_range': (226, 152027)
    },
    {
        'name': 'Motor Car (Jan 2026, normal)',
        'data': {
            'Year': 2026, 'Month': 'January', 'Standard_Category': 'Motor Car',
            'Prev_Month_New_Reg': 3251, 'Monthly_Growth_Rate': 4,
            'Month_Num': 1, 'Quarter': 1, 'Is_Peak_Season': 0, 'Is_Crisis_Period': 0,
            'Yearly_Total_Stock': 486196, 'Transfer': 9455,
            'Transfer_to_New_Ratio': 2.9, 'New_Registration_Market_Share': 0.12
        },
        'expected_range': (41, 21021)
    },
    {
        'name': 'Bus (Jan 2026, with 5% growth)',
        'data': {
            'Year': 2026, 'Month': 'January', 'Standard_Category': 'Bus',
            'Prev_Month_New_Reg': 118, 'Monthly_Growth_Rate': 5,
            'Month_Num': 1, 'Quarter': 1, 'Is_Peak_Season': 1, 'Is_Crisis_Period': 0,
            'Yearly_Total_Stock': 57859, 'Transfer': 1037,
            'Transfer_to_New_Ratio': 8.8, 'New_Registration_Market_Share': 0.01
        },
        'expected_range': (5, 578)
    }
]

print("=" * 70)
print("TESTING CORRECTED FORECASTS")
print("=" * 70)

for test in test_cases:
    try:
        response = requests.post('http://localhost:5000/predict', json=test['data'], timeout=5)
        result = response.json()
        if result.get('success'):
            pred = result.get('prediction', 0)
            min_val, max_val = test['expected_range']
            is_realistic = min_val <= pred <= max_val * 1.5
            status = "✓ OK" if is_realistic else "⚠ HIGH"
            print(f"\n{test['name']}:")
            print(f"  Forecast: {pred:.0f} vehicles")
            print(f"  Historical range: {min_val}-{max_val}")
            print(f"  Status: {status}")
        else:
            print(f"\n{test['name']}: ERROR - {result.get('error')}")
    except Exception as e:
        print(f"\n{test['name']}: Connection failed - {e}")

print("\n" + "=" * 70)
