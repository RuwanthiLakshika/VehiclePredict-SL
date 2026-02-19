import pandas as pd

df = pd.read_csv('data/master_dataset.csv')
print('=== FEATURE RANGES BY CATEGORY ===\n')
for category in sorted(df['Standard_Category'].unique()):
    cat_data = df[df['Standard_Category'] == category]
    print(f'{category}:')
    print(f'  New_Registration: min={cat_data["New_Registration"].min():.0f}, max={cat_data["New_Registration"].max():.0f}, mean={cat_data["New_Registration"].mean():.0f}')
    print(f'  Market_Share: min={cat_data["New_Registration_Market_Share"].min():.2f}, max={cat_data["New_Registration_Market_Share"].max():.2f}, mean={cat_data["New_Registration_Market_Share"].mean():.2f}')
    print(f'  Yearly_Stock: min={cat_data["Yearly_Total_Stock"].min():.0f}, max={cat_data["Yearly_Total_Stock"].max():.0f}')
    print(f'  Transfer: min={cat_data["Transfer"].min():.0f}, max={cat_data["Transfer"].max():.0f}')
    print(f'  Transfers_Ratio: min={cat_data["Transfer_to_New_Ratio"].min():.3f}, max={cat_data["Transfer_to_New_Ratio"].max():.3f}')
    print()
