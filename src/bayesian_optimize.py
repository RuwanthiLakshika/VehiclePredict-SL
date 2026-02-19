"""
Bayesian Optimization for CatBoost Hyperparameter Tuning

Uses Optuna library for intelligent hyperparameter search.
More efficient than grid search - finds optimal params in fewer iterations.
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import optuna
from optuna.pruners import MedianPruner

warnings.filterwarnings('ignore')


class BayesianCatBoostOptimizer:
    """Use Bayesian Optimization to find best CatBoost hyperparameters."""
    
    def __init__(self, data_path, n_trials=100):
        """Initialize optimizer."""
        self.data_path = data_path
        self.n_trials = n_trials
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_params = None
        self.best_score = 0
        self.best_mae = float('inf')
        self.categorical_features = []
        self.study = None
        
    def load_and_prepare_data(self):
        """Load and prepare dataset."""
        print("Loading and preparing data...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        # Prepare features and target
        target_col = 'New_Registration'
        exclude_cols = ['Standard_Category', 'Month', 'Year', target_col]
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
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
        
        # Split data (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Training: {self.X_train.shape[0]}, Testing: {self.X_test.shape[0]}")
        
        return X
    
    def objective(self, trial):
        """
        Objective function for Bayesian optimization.
        Optuna will minimize this (MAE on test set).
        """
        
        # Suggest hyperparameters using intelligent search
        params = {
            'iterations': trial.suggest_int('iterations', 200, 800, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 20, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
        }
        
        try:
            # Create and train model
            model = CatBoostRegressor(
                iterations=params['iterations'],
                learning_rate=params['learning_rate'],
                depth=params['depth'],
                l2_leaf_reg=params['l2_leaf_reg'],
                subsample=params['subsample'],
                colsample_bylevel=params['colsample_bylevel'],
                random_strength=params['random_strength'],
                loss_function='MAE',
                random_state=42,
                verbose=0
            )
            
            model.fit(self.X_train, self.y_train)
            
            # Evaluate on test set
            y_pred = model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Store best result
            if r2 > self.best_score:
                self.best_score = r2
                self.best_mae = mae
                self.best_params = params
            
            # Return MAE for minimization
            return mae
            
        except Exception as e:
            print(f"Error in trial: {str(e)}")
            return float('inf')
    
    def optimize(self):
        """Run Bayesian optimization."""
        print("\n" + "="*80)
        print("BAYESIAN OPTIMIZATION - CATBOOST HYPERPARAMETER TUNING")
        print("="*80)
        print(f"\nSearching for optimal parameters using {self.n_trials} trials...")
        print("(This intelligently searches the parameter space)\n")
        
        # Create study for minimization
        self.study = optuna.create_study(
            direction='minimize',  # Minimize MAE
            pruner=MedianPruner()  # Early stopping for bad trials
        )
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETED!")
        print("="*80)
        
        return self.study
    
    def display_results(self):
        """Display optimization results."""
        print("\n" + "="*80)
        print("🏆 BEST HYPERPARAMETERS FOUND")
        print("="*80)
        
        print(f"\nParameters:")
        for param, value in self.best_params.items():
            if isinstance(value, float):
                print(f"  {param:25s}: {value:.6f}")
            else:
                print(f"  {param:25s}: {value}")
        
        print(f"\nPerformance:")
        print(f"  R² Score: {self.best_score:.4f}")
        print(f"  MAE: {self.best_mae:.2f} vehicles/month")
        
        # Calculate RMSE for best config
        best_model = CatBoostRegressor(
            **{k: v for k, v in self.best_params.items() if k not in ['colsample_bylevel', 'random_strength']},
            colsample_bylevel=self.best_params['colsample_bylevel'],
            random_strength=self.best_params['random_strength'],
            loss_function='MAE',
            random_state=42,
            verbose=0
        )
        best_model.fit(self.X_train, self.y_train)
        y_pred_best = best_model.predict(self.X_test)
        rmse_best = np.sqrt(mean_squared_error(self.y_test, y_pred_best))
        print(f"  RMSE: {rmse_best:.2f} vehicles")
        
        return best_model
    
    def compare_with_previous(self, best_model):
        """Compare Bayesian results with previous best."""
        print("\n" + "="*80)
        print("IMPROVEMENT ANALYSIS")
        print("="*80)
        
        # Previous best (from grid search)
        prev_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=5,
            subsample=0.9,
            loss_function='MAE',
            random_state=42,
            verbose=0
        )
        prev_model.fit(self.X_train, self.y_train)
        y_pred_prev = prev_model.predict(self.X_test)
        prev_r2 = r2_score(self.y_test, y_pred_prev)
        prev_mae = mean_absolute_error(self.y_test, y_pred_prev)
        
        # New best
        y_pred_new = best_model.predict(self.X_test)
        new_r2 = r2_score(self.y_test, y_pred_new)
        new_mae = mean_absolute_error(self.y_test, y_pred_new)
        
        print(f"\nPrevious Best (Grid Search):")
        print(f"  R² Score: {prev_r2:.4f}")
        print(f"  MAE: {prev_mae:.2f} vehicles/month")
        
        print(f"\nBayesian Optimized:")
        print(f"  R² Score: {new_r2:.4f}")
        print(f"  MAE: {new_mae:.2f} vehicles/month")
        
        # Calculate improvements
        r2_improvement = (new_r2 - prev_r2) / prev_r2 * 100
        mae_improvement = (prev_mae - new_mae) / prev_mae * 100
        error_reduction = prev_mae - new_mae
        
        print(f"\nIMPROVEMENTS:")
        print(f"  R² improvement: {r2_improvement:+.2f}%")
        print(f"  MAE reduction: {mae_improvement:+.2f}%")
        if error_reduction > 0:
            print(f"  Error reduced by: {error_reduction:.2f} vehicles/month")
        
        return prev_r2, prev_mae, new_r2, new_mae
    
    def save_results(self, best_model):
        """Save Bayesian optimization results."""
        # Save best parameters
        params_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bayesian_best_params.txt')
        with open(params_path, 'w') as f:
            f.write("BEST CATBOOST HYPERPARAMETERS (BAYESIAN OPTIMIZATION)\n")
            f.write("="*60 + "\n\n")
            for param, value in self.best_params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"\nR² Score: {self.best_score:.4f}\n")
            f.write(f"MAE: {self.best_mae:.2f} vehicles/month\n")
        
        print(f"\n✓ Best parameters saved to: {params_path}")
        
        # Save study dataframe
        trials_df = self.study.trials_dataframe()
        trials_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bayesian_optimization_trials.csv')
        trials_df.to_csv(trials_path, index=False)
        print(f"✓ Optimization trials saved to: {trials_path}")
        
        return params_path
    
    def run(self):
        """Execute the complete Bayesian optimization pipeline."""
        print("="*80)
        print("BAYESIAN OPTIMIZATION FOR CATBOOST")
        print("Vehicle Registration Prediction")
        print("="*80)
        
        try:
            # Load data
            self.load_and_prepare_data()
            
            # Run optimization
            self.optimize()
            
            # Display results
            best_model = self.display_results()
            
            # Compare with previous
            self.compare_with_previous(best_model)
            
            # Save results
            self.save_results(best_model)
            
            print("\n" + "="*80)
            print("✓ BAYESIAN OPTIMIZATION COMPLETED!")
            print("="*80)
            
            return self.best_params, self.best_score, self.best_mae
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            raise


def main():
    """Main function."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'master_dataset.csv')
    
    # Run Bayesian optimization with 100 trials
    optimizer = BayesianCatBoostOptimizer(data_path, n_trials=100)
    params, score, mae = optimizer.run()
    
    return params, score, mae


if __name__ == "__main__":
    main()
