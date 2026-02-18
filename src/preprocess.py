"""
Preprocessing Script for RetailPredict Dataset

This script handles:
1. Data Integration: Merge 23 separate Excel files into unified dataset
2. Cleaning: Remove metadata noise and convert currency/text strings to floats
3. Imputation: Handle missing values ("-") with category-based averages
4. Feature Engineering: Apply categorization by market behavior (perishability)
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handle all preprocessing steps for the retail dataset."""
    
    def __init__(self, raw_data_path, output_path):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_path: Path to directory containing raw CSV files
            output_path: Path to save the master dataset
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.df = None
        
        # Define perishability categories for feature engineering
        self.perishability_mapping = {
            'perishable': ['fresh', 'dairy', 'meat', 'fish', 'fruits', 'vegetables', 'bakery'],
            'semi-perishable': ['frozen', 'refrigerated'],
            'non-perishable': ['canned', 'dry', 'beverages', 'snacks', 'grains', 'spices']
        }
    
    def load_and_merge_files(self):
        """
        Step 1: Data Integration
        Merge all CSV files from the raw data directory.
        
        Returns:
            pd.DataFrame: Merged dataset
        """
        print("Step 1: Data Integration - Loading and merging files...")
        
        # Find all CSV files in raw data directory
        csv_files = glob.glob(os.path.join(self.raw_data_path, '*.csv'))
        
        if not csv_files:
            print(f"Warning: No CSV files found in {self.raw_data_path}")
            print("Creating sample dataset for demonstration...")
            self.df = self._create_sample_data()
            return self.df
        
        dataframes = []
        for file_path in csv_files:
            try:
                df_temp = pd.read_csv(file_path)
                dataframes.append(df_temp)
                print(f"  Loaded: {os.path.basename(file_path)} - Shape: {df_temp.shape}")
            except Exception as e:
                print(f"  Error loading {file_path}: {str(e)}")
        
        # Concatenate all dataframes
        self.df = pd.concat(dataframes, ignore_index=True)
        
        print(f"✓ Merged dataset shape: {self.df.shape}")
        print(f"✓ Total unique products: {self.df['product_name'].nunique() if 'product_name' in self.df.columns else 'N/A'}")
        
        return self.df
    
    def clean_data(self):
        """
        Step 2: Data Cleaning
        Remove metadata noise and convert currency/text strings to floats.
        """
        print("\nStep 2: Data Cleaning...")
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_and_merge_files() first.")
        
        # Remove duplicate rows
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"  Removed duplicates: {initial_rows - len(self.df)} rows")
        
        # Remove rows where all values are NaN (except Product/Category)
        self.df = self.df.dropna(how='all', subset=[col for col in self.df.columns if col not in ['Product', 'Category']])
        
        # Rename columns for consistency
        if 'Product' in self.df.columns:
            self.df.rename(columns={'Product': 'product_name'}, inplace=True)
        if 'Category' in self.df.columns:
            self.df.rename(columns={'Category': 'category'}, inplace=True)
        
        # Handle currency/text to numeric conversion for all non-string columns
        for col in self.df.columns:
            if col not in ['product_name', 'category']:
                # Convert to numeric, treating '-', '', and NaN as missing
                self.df[col] = self.df[col].astype(str).str.strip()
                self.df[col] = self.df[col].replace(['', '-', 'nan'], np.nan)
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"✓ Data cleaning completed")
        missing_counts = self.df.isnull().sum()
        print(f"  Missing values by column:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"    {col}: {count}")
        
        return self.df
    
    def impute_missing_values(self):
        """
        Step 3: Missing Value Imputation
        Handle missing values ("-") with category-based averages.
        """
        print("\nStep 3: Imputation - Handling missing values...")
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_and_merge_files() first.")
        
        # Identify category column
        category_col = 'category' if 'category' in self.df.columns else None
        
        # Identify numeric columns for imputation (exclude product_name and category)
        numeric_cols = [col for col in self.df.columns if col not in ['product_name', 'category']]
        numeric_cols = [col for col in numeric_cols if self.df[col].dtype in [np.float64, np.int64, np.int32]]
        
        print(f"  Category column: {category_col}")
        print(f"  Numeric columns for imputation: {len(numeric_cols)}")
        
        # Impute missing values with category-based mean
        if category_col:
            for col in numeric_cols:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    # Calculate category-based mean
                    category_means = self.df.groupby(category_col)[col].transform('mean')
                    self.df[col].fillna(category_means, inplace=True)
                    
                    # Fill remaining NaN with overall mean
                    remaining_na = self.df[col].isnull().sum()
                    if remaining_na > 0:
                        overall_mean = self.df[col].mean()
                        self.df[col].fillna(overall_mean, inplace=True)
                        print(f"  Imputed {missing_count} missing values in '{col}'")
        else:
            # If no category column, use overall mean
            for col in numeric_cols:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    mean_val = self.df[col].mean()
                    self.df[col].fillna(mean_val, inplace=True)
                    print(f"  Imputed {missing_count} missing values in '{col}' with mean: {mean_val:.2f}")
        
        print(f"✓ Imputation completed")
        print(f"  Remaining missing values: {self.df.isnull().sum().sum()}")
        
        return self.df
    
    def feature_engineering(self):
        """
        Step 4: Feature Engineering
        Create categorization features based on price behavior and perishability.
        """
        print("\nStep 4: Feature Engineering...")
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_and_merge_files() first.")
        
        # Get all price columns (weeks and baseline)
        price_cols = [col for col in self.df.columns if col.startswith('W') or 'Price' in col or 'price' in col]
        
        # Create average price feature
        if price_cols:
            self.df['avg_price'] = self.df[price_cols].mean(axis=1)
            print(f"  Created 'avg_price' from {len(price_cols)} price columns")
        
        # Create price volatility feature (standard deviation of prices)
        if len(price_cols) > 1:
            self.df['price_volatility'] = self.df[price_cols].std(axis=1)
            print(f"  Created 'price_volatility' measure")
        
        # Create price range (max - min)
        if len(price_cols) > 1:
            self.df['price_range'] = self.df[price_cols].max(axis=1) - self.df[price_cols].min(axis=1)
            print(f"  Created 'price_range' feature")
        
        # Create price categories based on average price
        if 'avg_price' in self.df.columns:
            self.df['price_category'] = pd.qcut(
                self.df['avg_price'], 
                q=3, 
                labels=['low', 'medium', 'high'],
                duplicates='drop'
            )
            print(f"  Created 'price_category' (low, medium, high)")
        
        # Create perishability feature based on product category
        if 'category' in self.df.columns:
            def get_perishability(cat):
                cat_lower = str(cat).lower()
                if any(x in cat_lower for x in ['fruit', 'vegetable', 'dairy', 'meat', 'fish', 'egg', 'bakery', 'leafy']):
                    return 'perishable'
                elif any(x in cat_lower for x in ['frozen', 'refrigerated']):
                    return 'semi-perishable'
                else:
                    return 'non-perishable'
            
            self.df['perishability'] = self.df['category'].apply(get_perishability)
            print(f"  Created 'perishability' feature based on category")
        
        # Handle Change_Ann and Change_Wk if they exist
        if 'Change_Ann' in self.df.columns:
            self.df.rename(columns={'Change_Ann': 'annual_change'}, inplace=True)
        if 'Change_Wk' in self.df.columns:
            self.df.rename(columns={'Change_Wk': 'weekly_change'}, inplace=True)
        
        print(f"✓ Feature engineering completed")
        print(f"  New features created: {[col for col in self.df.columns if col in ['avg_price', 'price_volatility', 'price_range', 'price_category', 'perishability']]}")
        
        return self.df
    
    def _create_sample_data(self):
        """Create sample data for demonstration if no files are found."""
        print("  Creating sample dataset...")
        
        categories = ['Fresh Produce', 'Dairy', 'Meat & Fish', 'Beverages', 'Snacks', 
                     'Frozen Foods', 'Bakery', 'Canned Goods', 'Dry Goods', 'Spices', 'Other']
        
        np.random.seed(42)
        data = {
            'product_name': [f'Product_{i}' for i in range(122)],
            'category': np.random.choice(categories, 122),
            'price': np.random.uniform(1, 500, 122),
            'quantity': np.random.randint(1, 1000, 122),
            'expiry_days': np.random.choice([np.nan, 7, 14, 30, 60, 180, 365], 122),
            'rating': np.random.uniform(2, 5, 122)
        }
        
        return pd.DataFrame(data)
    
    def save_master_dataset(self):
        """Save the processed dataset to CSV."""
        print("\nSaving master dataset...")
        
        if self.df is None:
            raise ValueError("No data to save. Run preprocessing steps first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        self.df.to_csv(self.output_path, index=False)
        print(f"✓ Master dataset saved to: {self.output_path}")
        print(f"  Final shape: {self.df.shape}")
        print(f"  Columns: {list(self.df.columns)}")
        
        return self.output_path
    
    def get_summary_statistics(self):
        """Print summary statistics of the processed data."""
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"  Rows (Records): {self.df.shape[0]}")
        print(f"  Columns (Features): {self.df.shape[1]}")
        
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nNumerical Summary:")
        print(self.df.describe())
        
        print("\nCategorical Features:")
        for col in self.df.select_dtypes(include=['object']).columns:
            print(f"  {col}: {self.df[col].nunique()} unique values")
        
        print("="*60 + "\n")


def main():
    """Main preprocessing pipeline."""
    
    # Define paths
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'master_dataset.csv')
    
    print("="*60)
    print("RETAIL PREDICT - DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(raw_data_path, output_path)
    
    try:
        # Execute preprocessing steps in sequence
        preprocessor.load_and_merge_files()
        preprocessor.clean_data()
        preprocessor.impute_missing_values()
        preprocessor.feature_engineering()
        preprocessor.save_master_dataset()
        preprocessor.get_summary_statistics()
        
        print("\n✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n✗ ERROR during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
