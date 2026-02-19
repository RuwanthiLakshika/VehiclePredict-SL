"""
Preprocessing Script for Automotive Dataset - Sri Lanka

This script handles:
1. Data Load: Read automotive CSV dataset
2. Data Cleaning: Remove anomalies, handle missing values
3. Feature Engineering: Create derived metrics for model training
4. Data Validation: Ensure data quality and consistency

Source: Department of Motor Traffic (DMT), Sri Lanka Government
Data includes: New Vehicle Registrations, Transfers, Vehicle Population
"""

import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class AutomotiveDataPreprocessor:
    """Handle all preprocessing steps for the automotive dataset."""
    
    def __init__(self, raw_data_path, output_path):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_path: Path to raw automotive CSV file
            output_path: Path to save the processed master dataset
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.df = None
        self.df_processed = None
    
    def load_data(self):
        """
        Step 1: Data Loading
        Load the automotive dataset from CSV.
        
        Returns:
            pd.DataFrame: Loaded raw dataset
        """
        print("Step 1: Data Loading...")
        
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Dataset not found at {self.raw_data_path}")
        
        self.df = pd.read_csv(self.raw_data_path)
        print(f"✓ Dataset loaded: {self.df.shape}")
        print(f"  Columns: {list(self.df.columns)}")
        print(f"\n  Data Overview:")
        print(f"    Year range: {self.df['Year'].min()} - {self.df['Year'].max()}")
        print(f"    Categories: {self.df['Standard_Category'].unique().tolist()}")
        print(f"    Total records: {len(self.df)}")
        
        return self.df
    
    def clean_data(self):
        """
        Step 2: Data Cleaning
        Remove duplicates, handle missing values, validate data types.
        """
        print("\nStep 2: Data Cleaning...")
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Remove duplicate rows
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"  Removed duplicates: {initial_rows - len(self.df)} rows")
        
        # Check for missing values
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"  Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"    {col}: {count}")
        else:
            print(f"  ✓ No missing values")
        
        # Ensure numeric columns are proper types
        numeric_cols = ['Year', 'Month_Num', 'Quarter', 'New_Registration', 
                       'Transfer', 'Yearly_Total_Stock', 'Prev_Month_New_Reg']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Remove any rows with critical missing values
        self.df = self.df.dropna(subset=['Year', 'Month_Num', 'New_Registration'])
        
        print(f"✓ Data cleaning completed. Final shape: {self.df.shape}")
        
        return self.df
    
    def validate_data(self):
        """
        Step 3: Data Validation
        Validate data consistency and logical constraints.
        """
        print("\nStep 3: Data Validation...")
        
        # Check year range
        year_min, year_max = self.df['Year'].min(), self.df['Year'].max()
        print(f"  Year range: {year_min} - {year_max}")
        
        # Check month numbers
        month_range = self.df['Month_Num'].unique()
        if set(month_range) <= set(range(1, 13)):
            print(f"  ✓ Month numbers valid: {sorted(month_range)}")
        else:
            print(f"  ✗ Invalid month numbers found: {month_range}")
        
        # Check positive values
        cols_positive = ['New_Registration', 'Transfer', 'Yearly_Total_Stock']
        for col in cols_positive:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    print(f"  ✗ {col}: {negative_count} negative values found")
                else:
                    print(f"  ✓ {col}: All values positive")
        
        # Check for outliers
        print(f"\n  Value Ranges:")
        print(f"    New_Registration: {self.df['New_Registration'].min():.0f} - {self.df['New_Registration'].max():.0f}")
        print(f"    Transfer: {self.df['Transfer'].min():.0f} - {self.df['Transfer'].max():.0f}")
        print(f"    Yearly_Total_Stock: {self.df['Yearly_Total_Stock'].min():.0f} - {self.df['Yearly_Total_Stock'].max():.0f}")
        
        return self.df
    
    def engineer_features(self):
        """
        Step 4: Feature Engineering
        Create derived features for enhanced model training.
        """
        print("\nStep 4: Feature Engineering...")
        
        # Create month name if not present
        if 'Month' not in self.df.columns:
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            self.df['Month'] = self.df['Month_Num'].map(month_names)
        
        # Seasonality features (already present as Is_Peak_Season, Is_Crisis_Period)
        # Verify they exist, if not create them
        if 'Is_Peak_Season' not in self.df.columns:
            # Peak season: April, May, November, December
            peak_months = [4, 5, 11, 12]
            self.df['Is_Peak_Season'] = self.df['Month_Num'].isin(peak_months).astype(int)
        
        if 'Is_Crisis_Period' not in self.df.columns:
            # Crisis periods: 2022-2023 (economic crisis)
            self.df['Is_Crisis_Period'] = ((self.df['Year'] >= 2022) & (self.df['Year'] <= 2023)).astype(int)
        
        # Ratio features (verify existence)
        if 'Transfer_to_New_Ratio' not in self.df.columns:
            self.df['Transfer_to_New_Ratio'] = self.df['Transfer'] / (self.df['New_Registration'] + 1)
        
        if 'New_Registration_Market_Share' not in self.df.columns:
            # Market share by category per month
            self.df = self.df.sort_values(['Year', 'Month_Num'])
            monthly_totals = self.df.groupby(['Year', 'Month_Num'])['New_Registration'].transform('sum')
            self.df['New_Registration_Market_Share'] = self.df['New_Registration'] / (monthly_totals + 1)
        
        if 'Monthly_Growth_Rate' not in self.df.columns:
            # Month-over-month growth rate
            self.df = self.df.sort_values(['Standard_Category', 'Year', 'Month_Num'])
            self.df['Monthly_Growth_Rate'] = self.df.groupby('Standard_Category')['New_Registration'].pct_change()
        
        if 'Prev_Month_New_Reg' not in self.df.columns:
            # Previous month's registrations
            self.df['Prev_Month_New_Reg'] = self.df.groupby('Standard_Category')['New_Registration'].shift(1)
        
        # Fill NaN in derived features
        self.df['Monthly_Growth_Rate'].fillna(0, inplace=True)
        self.df['Prev_Month_New_Reg'].fillna(0, inplace=True)
        
        print(f"✓ Feature engineering completed")
        print(f"  Created/Verified features:")
        print(f"    - Seasonality: Is_Peak_Season, Is_Crisis_Period")
        print(f"    - Ratios: Transfer_to_New_Ratio, New_Registration_Market_Share")
        print(f"    - Trends: Monthly_Growth_Rate, Prev_Month_New_Reg")
        print(f"  Final shape: {self.df.shape}")
        
        return self.df
    
    def generate_statistics(self):
        """
        Step 5: Generate Descriptive Statistics
        Provide insights into the data distribution.
        """
        print("\nStep 5: Data Statistics and Insights...")
        
        print(f"\n  New Registrations by Category:")
        category_stats = self.df.groupby('Standard_Category')['New_Registration'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(category_stats.round(2))
        
        print(f"\n  Vehicle Category Distribution:")
        print(f"    Categories: {sorted(self.df['Standard_Category'].unique())}")
        print(f"    Total categories: {self.df['Standard_Category'].nunique()}")
        
        print(f"\n  Year Coverage:")
        years = sorted(self.df['Year'].unique())
        print(f"    Years: {years}")
        print(f"    Records per year: {self.df['Year'].value_counts().sort_index().to_dict()}")
        
        return self.df
    
    def save_processed_data(self):
        """
        Step 6: Save Processed Data
        Save the cleaned and engineered dataset to CSV.
        """
        print("\nStep 6: Saving Processed Data...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Save to CSV
        self.df.to_csv(self.output_path, index=False)
        print(f"✓ Processed dataset saved to: {self.output_path}")
        print(f"  Final dataset shape: {self.df.shape}")
        print(f"  Columns: {list(self.df.columns)}")
        
        # Save summary statistics
        summary_path = self.output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("AUTOMOTIVE DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset Shape: {self.df.shape}\n\n")
            f.write(f"Columns:\n{chr(10).join(['  - ' + col for col in self.df.columns])}\n\n")
            f.write(f"Data Types:\n{self.df.dtypes.to_string()}\n\n")
            f.write(f"Missing Values:\n{self.df.isnull().sum().to_string()}\n\n")
            f.write(f"Statistical Summary:\n{self.df.describe().to_string()}\n")
        
        print(f"✓ Summary statistics saved to: {summary_path}")
        
        return self.df
    
    def preprocess_pipeline(self):
        """Execute the complete preprocessing pipeline."""
        print("=" * 60)
        print("AUTOMOTIVE DATASET PREPROCESSING PIPELINE")
        print("=" * 60)
        
        try:
            # Execute pipeline steps
            self.load_data()
            self.clean_data()
            self.validate_data()
            self.engineer_features()
            self.generate_statistics()
            self.save_processed_data()
            
            print("\n" + "=" * 60)
            print("✓ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"\nDataset Ready for Model Training:")
            print(f"  Path: {self.output_path}")
            print(f"  Shape: {self.df.shape}")
            print(f"  Features: {len(self.df.columns)}")
            
            return self.df
            
        except Exception as e:
            print(f"\n✗ ERROR during preprocessing: {str(e)}")
            raise


def main():
    """Main function to run the preprocessing pipeline."""
    
    # Define paths
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'SriLanka_Automotive_Advanced_Features.csv')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'master_dataset.csv')
    
    # Create preprocessor and run pipeline
    preprocessor = AutomotiveDataPreprocessor(raw_data_path, output_path)
    df_processed = preprocessor.preprocess_pipeline()
    
    return df_processed


if __name__ == "__main__":
    main()
