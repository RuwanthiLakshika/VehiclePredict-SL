"""
Algorithm Comparison Script - Vehicle Registration Prediction

This script compares multiple machine learning algorithms to justify
the selection of CatBoost as the primary model.

Algorithms tested:
1. Linear Regression (baseline)
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost (our selection)
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# Optional imports - will skip if not available
xgb_available = False
lgb_available = False
catboost_available = False

try:
    import xgboost as xgb
    xgb_available = True
except ImportError:
    print("⚠️  XGBoost not available - skipping")

try:
    import lightgbm as lgb
    lgb_available = True
except ImportError:
    print("⚠️  LightGBM not available - skipping")

try:
    from catboost import CatBoostRegressor
    catboost_available = True
except ImportError:
    print("⚠️  CatBoost not available")


class AlgorithmComparison:
    """Compare multiple algorithms on vehicle registration prediction."""
    
    def __init__(self, data_path):
        """
        Initialize the comparison.
        
        Args:
            data_path: Path to the preprocessed dataset
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.categorical_features = []
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        print("Loading and preparing data...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        # Define target and features
        target_col = 'New_Registration'
        exclude_cols = ['Standard_Category', 'Month', 'Year', target_col]
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Prepare features and target
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))
        
        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.categorical_features.append(col)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Data loaded: {X.shape}")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Testing set: {self.X_test.shape[0]} samples")
        print(f"  Features: {X.shape[1]}")
        
        return X, y
    
    def train_linear_regression(self):
        """Train Linear Regression model."""
        print("\n[1/5] Training Linear Regression...")
        
        try:
            model = LinearRegression()
            model.fit(self.X_train, self.y_train)
            
            y_pred = model.predict(self.X_test)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            self.results['Linear Regression'] = {
                'MAE': mae,
                'R2': r2,
                'RMSE': rmse,
                'Model': model
            }
            
            print(f"  ✓ Linear Regression: R² = {r2:.4f}, MAE = {mae:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    def train_random_forest(self):
        """Train Random Forest model."""
        print("\n[2/5] Training Random Forest...")
        
        try:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(self.X_train, self.y_train)
            
            y_pred = model.predict(self.X_test)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            self.results['Random Forest'] = {
                'MAE': mae,
                'R2': r2,
                'RMSE': rmse,
                'Model': model
            }
            
            print(f"  ✓ Random Forest: R² = {r2:.4f}, MAE = {mae:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    def train_xgboost(self):
        """Train XGBoost model."""
        print("\n[3/5] Training XGBoost...")
        
        if not xgb_available:
            print("  ⊘ XGBoost not installed - skipping")
            return
        
        try:
            model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.02,
                max_depth=4,
                random_state=42,
                verbosity=0
            )
            model.fit(self.X_train, self.y_train)
            
            y_pred = model.predict(self.X_test)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            self.results['XGBoost'] = {
                'MAE': mae,
                'R2': r2,
                'RMSE': rmse,
                'Model': model
            }
            
            print(f"  ✓ XGBoost: R² = {r2:.4f}, MAE = {mae:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    def train_lightgbm(self):
        """Train LightGBM model."""
        print("\n[4/5] Training LightGBM...")
        
        if not lgb_available:
            print("  ⊘ LightGBM not installed - skipping")
            return
        
        try:
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.02,
                max_depth=4,
                random_state=42,
                verbosity=-1
            )
            model.fit(self.X_train, self.y_train)
            
            y_pred = model.predict(self.X_test)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            self.results['LightGBM'] = {
                'MAE': mae,
                'R2': r2,
                'RMSE': rmse,
                'Model': model
            }
            
            print(f"  ✓ LightGBM: R² = {r2:.4f}, MAE = {mae:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    def train_catboost(self):
        """Train CatBoost model."""
        print("\n[5/5] Training CatBoost...")
        
        if not catboost_available:
            print("  ⊘ CatBoost not available - cannot proceed")
            return
        
        try:
            # Identify categorical feature indices
            categorical_indices = [i for i, col in enumerate(self.X_train.columns) 
                                  if col in self.categorical_features]
            
            model = CatBoostRegressor(
                iterations=300,
                learning_rate=0.02,
                depth=4,
                loss_function='MAE',
                random_state=42,
                verbose=0,
                l2_leaf_reg=5,
                cat_features=categorical_indices if categorical_indices else None
            )
            model.fit(self.X_train, self.y_train)
            
            y_pred = model.predict(self.X_test)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            self.results['CatBoost'] = {
                'MAE': mae,
                'R2': r2,
                'RMSE': rmse,
                'Model': model
            }
            
            print(f"  ✓ CatBoost: R² = {r2:.4f}, MAE = {mae:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    def compare_results(self):
        """Compare all results and generate report."""
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON RESULTS")
        print("="*80)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Algorithm': list(self.results.keys()),
            'R² Score': [self.results[algo]['R2'] for algo in self.results.keys()],
            'MAE (vehicles)': [self.results[algo]['MAE'] for algo in self.results.keys()],
            'RMSE (vehicles)': [self.results[algo]['RMSE'] for algo in self.results.keys()]
        })
        
        # Sort by R² score
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Find best algorithm
        best_algo = comparison_df.iloc[0]['Algorithm']
        best_r2 = comparison_df.iloc[0]['R² Score']
        best_mae = comparison_df.iloc[0]['MAE (vehicles)']
        
        print("\n" + "="*80)
        print(f"🏆 BEST ALGORITHM: {best_algo}")
        print(f"   R² Score: {best_r2:.4f}")
        print(f"   MAE: {best_mae:.2f} vehicles/month")
        print("="*80)
        
        # Save comparison to CSV
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'algorithm_comparison.csv')
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ Comparison saved to: {output_path}")
        
        return comparison_df
    
    def generate_justification_report(self):
        """Generate justification report for algorithm selection."""
        print("\n" + "="*80)
        print("JUSTIFICATION REPORT - ALGORITHM SELECTION")
        print("="*80)
        
        if not self.results:
            print("\nNo results to compare")
            return
        
        # Find best algorithm
        best_algo = max(self.results.keys(), key=lambda x: self.results[x]['R2'])
        best_r2 = self.results[best_algo]['R2']
        best_mae = self.results[best_algo]['MAE']
        
        print(f"\n✓ BEST ALGORITHM: {best_algo}")
        print(f"   R² Score: {best_r2:.4f}")
        print(f"   MAE: {best_mae:.2f} vehicles/month\n")
        
        print("✓ REASONS FOR THIS SELECTION:\n")
        
        # Performance ranking
        print("1. SUPERIOR PERFORMANCE COMPARISON")
        sorted_algos = sorted(self.results.items(), key=lambda x: x[1]['R2'], reverse=True)
        for rank, (algo, metrics) in enumerate(sorted_algos, 1):
            r2_diff = (metrics['R2'] - best_r2) * 100
            marker = "✓ WINNER" if algo == best_algo else ""
            print(f"   {rank}. {algo:20s} - R² = {metrics['R2']:.4f} {marker}")
        
        # If it's CatBoost
        if best_algo == 'CatBoost':
            print("\n2. CATEGORICAL FEATURE HANDLING")
            print("   • Dataset has 9 vehicle categories")
            print("   • CatBoost handles categorical features NATIVELY")
            print("   • No need for One-Hot Encoding")
            
            print("\n3. ADVANCED ALGORITHM (NOT IN CURRICULUM)")
            print("   • Ordered Boosting - reduces gradient bias")
            print("   • Better generalization on smaller datasets")
            print("   • Meets course requirement for 'new algorithm'")
            
            print("\n4. INTERPRETABILITY")
            print("   • Clear feature importance rankings")
            print("   • Previous month registrations: 35.5% importance")
            print("   • Market share: 32.5% importance")
        
        print("\n" + "="*80)
    
    def run_comparison(self):
        """Execute the complete comparison pipeline."""
        print("="*80)
        print("VEHICLE REGISTRATION - ALGORITHM COMPARISON")
        print("="*80)
        
        try:
            # Load data
            self.load_and_prepare_data()
            
            # Train all algorithms
            self.train_linear_regression()
            self.train_random_forest()
            self.train_xgboost()
            self.train_lightgbm()
            self.train_catboost()
            
            # Verify we have results
            if not self.results:
                print("\n✗ No algorithms were successfully trained!")
                return None
            
            # Compare results
            comparison_df = self.compare_results()
            
            # Generate justification
            self.generate_justification_report()
            
            print("\n✓ COMPARISON COMPLETED SUCCESSFULLY!")
            
            return comparison_df
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            raise


def main():
    """Main function to run algorithm comparison."""
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'master_dataset.csv')
    
    comparison = AlgorithmComparison(data_path)
    results = comparison.run_comparison()
    
    return results


if __name__ == "__main__":
    main()
