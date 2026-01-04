"""
Data loading utilities for time series phenotyping.
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Loads data for time series phenotyping analysis.
    
    Example:
        loader = DataLoader()
        df = loader.load_analysis_data('data/cohort_analysis.csv')
    """
    
    def __init__(self, label_col='pred', patient_col='pat'):
        """
        Initialize the data loader.
        
        Args:
            label_col: Column name for the prediction/label column.
            patient_col: Column name for patient identifiers.
        """
        self.label_col = label_col
        self.patient_col = patient_col
    
    def load_analysis_data(self, filepath, positive_only=False):
        """
        Load the cohort analysis DataFrame.
        
        Args:
            filepath: Path to the CSV file.
            positive_only: If True, keep only positive label predictions.
            
        Returns:
            Loaded DataFrame.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0)
        
        if positive_only and self.label_col in df.columns:
            df = df[df[self.label_col] == True]
        
        print(f"Loaded {len(df)} records from {filepath.name}")
        
        return df
    
    def load_burden_profiles(self, filepath):
        """
        Load pre-computed burden profiles.
        
        Args:
            filepath: Path to the burden profiles CSV file.
            
        Returns:
            DataFrame with burden profiles (patients as rows, time bins as columns).
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0)
        
        print(f"Loaded burden profiles: {df.shape[0]} patients, {df.shape[1]} time bins")
        
        return df
    
    def load_chronophenotypes(self, filepath):
        """
        Load chronophenotypes file (burden profiles + cluster labels).
        
        Args:
            filepath: Path to the chronophenotypes CSV file.
            
        Returns:
            DataFrame with burden profiles and cluster column.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0)
        
        if 'cluster' not in df.columns:
            raise ValueError("Chronophenotypes file must have a 'cluster' column")
        
        n_clusters = df['cluster'].nunique()
        print(f"Loaded chronophenotypes: {df.shape[0]} patients, {n_clusters} clusters")
        
        return df
    
    def save_results(self, df, filepath):
        """
        Save results DataFrame to CSV.
        
        Args:
            df: DataFrame to save.
            filepath: Output path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        
        print(f"Results saved to {filepath}")
