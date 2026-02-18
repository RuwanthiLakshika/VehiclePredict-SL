"""
Training Script for CatBoost Model - Retail Price Prediction

This script:
1. Loads and prepares the preprocessed dataset
2. Splits data into 80/20 train-test sets
3. Trains a CatBoost model with optimized hyperparameters
4. Evaluates performance using MAE and R² metrics
5. Saves the trained model
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostRegressor
except ImportError:
    print("CatBoost not installed. Installing...")
    os.system('pip install catboost')
    from catboost import CatBoostRegressor


class RetailPriceModelTrainer:
    """Train and evaluate CatBoost model for retail price prediction."""
    
    def __init__(self, data_path, model_path):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the preprocessed master dataset
            model_path: Path to save the trained model
        """
        self.data_path = data_path
        self.model_path = model_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.predictions = None
        self.metrics = {}
        
    def load_data(self):
        """Load the preprocessed dataset."""
        print("Step 1: Loading Data...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Dataset loaded: {self.df.shape}")
        print(f"  Columns: {list(self.df.columns)}")
        
        return self.df
    
    def prepare_features_and_target(self):
        """
        Prepare features and target variable.
        
        Target: W4_Jan_2026 (final week price)
        Features: All other numeric and categorical columns
        """
        print("\nStep 2: Preparing Features and Target...")
        
        # Define target variable
        target_col = 'W4_Jan_2026'
        
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Remove rows with missing target values
        df_clean = self.df.dropna(subset=[target_col])
        print(f"  Removed rows with missing target: {len(self.df) - len(df_clean)}")
        
        # Define features (exclude product_name and target)
        exclude_cols = ['product_name', target_col]
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Handle remaining missing values
        X = X.fillna(X.mean(numeric_only=True))
        
        print(f"✓ Features prepared: {X.shape[1]} features, {len(X)} samples")
        print(f"  Feature columns: {feature_cols}")
        print(f"  Target variable: {target_col}")
        print(f"  Target statistics:")
        print(f"    Mean: {y.mean():.2f}")
        print(f"    Std: {y.std():.2f}")
        print(f"    Min: {y.min():.2f}")
        print(f"    Max: {y.max():.2f}")
        
        return X, y, feature_cols
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set (0.2 = 20%)
            random_state: Random seed for reproducibility
        """
        print("\nStep 3: Data Splitting...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✓ Data split completed:")
        print(f"  Training set: {self.X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"  Testing set: {self.X_test.shape[0]} samples ({test_size*100:.0f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def identify_categorical_features(self, X):
        """
        Identify categorical columns for CatBoost.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of categorical column indices
        """
        categorical_features = []
        
        for i, col in enumerate(X.columns):
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(i)
                print(f"  Identified categorical feature: {col}")
        
        return categorical_features
    
    def train_model(self, learning_rate=0.05, iterations=1000, depth=6):
        """
        Train CatBoost model with specified hyperparameters.
        
        Args:
            learning_rate: Learning rate (default: 0.05)
            iterations: Number of boosting iterations (default: 1000)
            depth: Tree depth (default: 6)
        """
        print("\nStep 4: Training CatBoost Model...")
        print(f"  Hyperparameters:")
        print(f"    Learning Rate: {learning_rate}")
        print(f"    Iterations: {iterations}")
        print(f"    Tree Depth: {depth}")
        
        # Identify categorical features
        categorical_features = self.identify_categorical_features(self.X_train)
        
        # Initialize CatBoost model
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='MAE',
            verbose=100,  # Print progress every 100 iterations
            random_state=42,
            cat_features=categorical_features if categorical_features else None
        )
        
        # Train the model
        print("\n  Training in progress...")
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            use_best_model=True
        )
        
        print("✓ Model training completed")
        
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate model performance on train and test sets.
        
        Returns:
            Dictionary containing all metrics
        """
        print("\nStep 5: Model Evaluation...")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        self.predictions = y_test_pred
        
        # Calculate metrics for training set
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        # Calculate metrics for test set
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        # Store metrics
        self.metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }
        
        # Print results
        print("✓ Evaluation Results:")
        print("\n  Training Set Metrics:")
        print(f"    MAE (Mean Absolute Error): {train_mae:.4f} LKR")
        print(f"    RMSE (Root Mean Squared Error): {train_rmse:.4f} LKR")
        print(f"    R² (Coefficient of Determination): {train_r2:.4f}")
        
        print("\n  Testing Set Metrics:")
        print(f"    MAE (Mean Absolute Error): {test_mae:.4f} LKR")
        print(f"    RMSE (Root Mean Squared Error): {test_rmse:.4f} LKR")
        print(f"    R² (Coefficient of Determination): {test_r2:.4f}")
        
        print("\n  Interpretation:")
        print(f"    The model explains {test_r2*100:.2f}% of the variance in test prices.")
        print(f"    On average, predictions deviate by ±{test_mae:.2f} LKR from actual prices.")
        
        return self.metrics
    
    def get_feature_importance(self, top_n=15):
        """
        Get feature importance from the trained model.
        
        Args:
            top_n: Number of top features to display
            
        Returns:
            DataFrame with feature importance
        """
        print("\nStep 6: Feature Importance Analysis...")
        
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        for idx, row in feature_importance.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    
    def save_model(self):
        """Save the trained model to disk."""
        print("\nStep 7: Saving Model...")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save using CatBoost's native format
        self.model.save_model(self.model_path)
        print(f"✓ Model saved to: {self.model_path}")
        
        return self.model_path
    
    def train_pipeline(self):
        """Execute the complete training pipeline."""
        print("="*60)
        print("CATBOOST MODEL TRAINING PIPELINE")
        print("="*60)
        
        try:
            # Load and prepare data
            self.load_data()
            X, y, feature_cols = self.prepare_features_and_target()
            
            # Split data
            self.split_data(X, y)
            
            # Train model
            self.train_model(learning_rate=0.05, iterations=1000, depth=6)
            
            # Evaluate model
            self.evaluate_model()
            
            # Feature importance
            feature_importance = self.get_feature_importance()
            
            # Save model
            self.save_model()
            
            print("\n" + "="*60)
            print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return {
                'model': self.model,
                'metrics': self.metrics,
                'feature_importance': feature_importance,
                'x_test': self.X_test,
                'y_test': self.y_test,
                'predictions': self.predictions
            }
            
        except Exception as e:
            print(f"\n✗ ERROR during training: {str(e)}")
            raise


def main():
    """Main function to run the training pipeline."""
    
    # Define paths
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'master_dataset.csv')
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'catboost_model.bin')
    
    # Create trainer and run pipeline
    trainer = RetailPriceModelTrainer(data_path, model_path)
    results = trainer.train_pipeline()
    
    return results


if __name__ == "__main__":
    main()
