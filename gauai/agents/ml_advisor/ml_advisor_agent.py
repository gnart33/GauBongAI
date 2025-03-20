from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class MLAdvisorAgent:
    """Agent that provides ML-related recommendations and insights."""
    
    def __init__(self):
        self.task_recommendations = {
            "regression": {
                "algorithms": ["linear_regression", "random_forest", "xgboost"],
                "metrics": ["mse", "rmse", "mae", "r2"]
            },
            "classification": {
                "algorithms": ["logistic_regression", "random_forest", "xgboost"],
                "metrics": ["accuracy", "precision", "recall", "f1"]
            },
            "clustering": {
                "algorithms": ["kmeans", "dbscan", "hierarchical"],
                "metrics": ["silhouette", "calinski_harabasz", "davies_bouldin"]
            }
        }
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Analyze dataset and provide ML recommendations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing analysis results and recommendations
        """
        analysis = {
            "data_characteristics": self._analyze_data_characteristics(df),
            "feature_recommendations": self._recommend_feature_engineering(df),
            "task_suggestions": self._suggest_ml_tasks(df),
            "preprocessing_steps": self._suggest_preprocessing(df)
        }
        return analysis
    
    def _analyze_data_characteristics(self, df: pd.DataFrame) -> Dict:
        """Analyze basic characteristics of the dataset."""
        return {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "feature_types": df.dtypes.value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
    
    def _recommend_feature_engineering(self, df: pd.DataFrame) -> List[Dict]:
        """Suggest feature engineering steps."""
        recommendations = []
        
        # Categorical features
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            recommendations.append({
                "type": "categorical_encoding",
                "columns": list(cat_cols),
                "methods": ["one_hot", "label", "target"]
            })
            
        # Numeric features
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            recommendations.append({
                "type": "scaling",
                "columns": list(num_cols),
                "methods": ["standard", "minmax", "robust"]
            })
            
        return recommendations
    
    def _suggest_ml_tasks(self, df: pd.DataFrame) -> List[Dict]:
        """Suggest potential ML tasks based on data characteristics."""
        suggestions = []
        
        # Check for potential regression tasks
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            suggestions.append({
                "task_type": "regression",
                "potential_targets": list(num_cols),
                "algorithms": self.task_recommendations["regression"]
            })
            
        # Check for potential classification tasks
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            suggestions.append({
                "task_type": "classification",
                "potential_targets": list(cat_cols),
                "algorithms": self.task_recommendations["classification"]
            })
            
        return suggestions
    
    def _suggest_preprocessing(self, df: pd.DataFrame) -> List[Dict]:
        """Suggest preprocessing steps."""
        suggestions = []
        
        # Missing values
        if df.isnull().any().any():
            suggestions.append({
                "type": "missing_values",
                "methods": ["mean", "median", "mode", "knn_imputation"]
            })
            
        # Outlier handling
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            suggestions.append({
                "type": "outlier_handling",
                "methods": ["iqr", "zscore", "isolation_forest"]
            })
            
        return suggestions 