from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats

class StatisticalAnalyzer:
    """Performs statistical analysis on datasets."""
    
    def __init__(self):
        self.results_cache = {}
        
    def describe_dataset(
        self,
        df: pd.DataFrame,
        include: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate comprehensive descriptive statistics.
        
        Args:
            df: Input DataFrame
            include: Types of columns to include in analysis
            
        Returns:
            Dict containing various statistical measures
        """
        stats_dict = {
            "basic_stats": df.describe(include=include).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_counts": df.nunique().to_dict()
        }
        
        # Add correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            stats_dict["correlations"] = df[numeric_cols].corr().to_dict()
            
        return stats_dict
    
    def test_normality(
        self,
        data: Union[pd.Series, np.ndarray],
        test_method: str = "shapiro"
    ) -> Dict:
        """
        Perform normality test on data.
        
        Args:
            data: Input data
            test_method: Statistical test to use ('shapiro' or 'kstest')
            
        Returns:
            Dict containing test results
        """
        if test_method == "shapiro":
            statistic, p_value = stats.shapiro(data)
        else:
            statistic, p_value = stats.kstest(data, "norm")
            
        return {
            "test_method": test_method,
            "statistic": statistic,
            "p_value": p_value,
            "is_normal": p_value > 0.05
        }
    
    def detect_outliers(
        self,
        data: Union[pd.Series, np.ndarray],
        method: str = "iqr"
    ) -> Dict:
        """
        Detect outliers in the data.
        
        Args:
            data: Input data
            method: Method to use for outlier detection ('iqr' or 'zscore')
            
        Returns:
            Dict containing outlier information
        """
        if method == "iqr":
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        else:
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
            
        return {
            "method": method,
            "n_outliers": len(outliers),
            "outlier_indices": list(outliers.index) if hasattr(outliers, 'index') else None,
            "outlier_values": list(outliers)
        } 