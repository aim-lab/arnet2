"""
Data preprocessing for time series phenotyping.

Transforms raw event data into time-binned burden profiles.
"""

import numpy as np
import pandas as pd

# Time constants
N_SECONDS_IN_HOUR = 3600
N_HOURS_IN_DAY = 24


class ShortBurdenPreprocessor:
    """
    Preprocesses time series data into short-term burden profiles (1-hour bins).
    
    The output is a matrix where:
    - Rows = patients/subjects
    - Columns = time bins (e.g., 24 hours)
    - Values = fraction of positive events in each 1-hour time bin

    Example:
        preprocessor = ShortBurdenPreprocessor()
        burden_profiles = preprocessor.compute_burden_profiles(df_all, df_positive)
    """
    
    def __init__(self, time_col='start_time', patient_col='pat', n_bins=24):
        """
        Initialize the preprocessor.
        
        Args:
            time_col: Column name for event start times (**in seconds**).
            patient_col: Column name for patient/subject identifiers.
            n_bins: Number of time bins (default 24 for hourly bins).
        """
        self.time_col = time_col
        self.patient_col = patient_col
        self.n_bins = n_bins
    
    def create_time_bins(self, df):
        """
        Create time-binned event counts per patient.
        
        Args:
            df: DataFrame with event data.
            
        Returns:
            Tuple of (per_patient_bins, population_bins) DataFrames.
        """
        # Create copy with time bin info
        binned = df[[self.patient_col, self.time_col]].copy()
        
        # Convert time to hour of day (0-23)
        binned['time_bin'] = (
            (binned[self.time_col] / N_SECONDS_IN_HOUR).astype(int) % self.n_bins
        )
        
        # Count events per patient per time bin
        per_patient = (
            binned
            .groupby([self.patient_col, 'time_bin'])
            .size()
            .reset_index(name='event_count')
        )
        
        # Count total events per time bin (population level)
        population = (
            binned
            .groupby('time_bin')
            .size()
            .reset_index(name='event_count')
        )
        
        return per_patient, population
    
    def compute_burden_profiles(self, df_all, df_positive):
        """
        Compute the burden profile matrix.
        
        Each row is a patient, each column is an hour (0-23), 
        and values are the fraction of positive events in that hour.
        
        Args:
            df_all: DataFrame with all events.
            df_positive: DataFrame with only positive label events.
            
        Returns:
            DataFrame with burden profiles (patients x time bins).
        """
        # Get binned counts
        per_patient_all, _ = self.create_time_bins(df_all)
        per_patient_pos, _ = self.create_time_bins(df_positive)
        
        # Rename columns for merge
        per_patient_all = per_patient_all.rename(columns={'event_count': 'total_count'})
        per_patient_pos = per_patient_pos.rename(columns={'event_count': 'positive_count'})
        
        # Merge positive and total counts
        merged = per_patient_pos.merge(
            per_patient_all,
            on=[self.patient_col, 'time_bin'],
            how='outer'
        ).fillna(0)
        
        # Pivot to patient x time_bin matrix
        pivot_pos = merged.pivot(
            index=self.patient_col,
            columns='time_bin',
            values='positive_count'
        ).fillna(0)
        
        pivot_total = merged.pivot(
            index=self.patient_col,
            columns='time_bin',
            values='total_count'
        ).fillna(0)
        
        # Compute burden fraction
        burden_matrix = pivot_pos / pivot_total.replace(0, np.nan)
        
        # Fill NaN with row median
        row_medians = burden_matrix.median(axis=1)
        for idx in burden_matrix.index:
            burden_matrix.loc[idx] = burden_matrix.loc[idx].fillna(row_medians[idx])
        
        print(f"Computed burden profiles: {burden_matrix.shape[0]} subjects, {burden_matrix.shape[1]} time bins")
        
        return burden_matrix
    
    def compute_mean_burden(self, burden_profiles):
        """
        Compute mean burden per patient.
        
        Args:
            burden_profiles: Patient x time bin burden matrix.
            
        Returns:
            Series with mean burden per patient.
        """
        return burden_profiles.mean(axis=1)


if __name__ == '__main__':
    # Example usage
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_events = 500
    
    # Simulate data: patients with windows at different times, each labeled AF (1) or non-AF (0)
    df = pd.DataFrame({
        'pat': np.repeat(['P001', 'P002', 'P003', 'P004', 'P005'], n_events // 5),
        'start_time': np.random.uniform(0, 86400, n_events),  # random time of day (seconds)
        'pred': np.random.choice([True, False], n_events, p=[0.3, 0.7])  # 30% AF
    })
    
    # Compute burden profiles
    preprocessor = ShortBurdenPreprocessor()
    df_positive = df[df['pred'] == True]
    burden_profiles = preprocessor.compute_burden_profiles(df, df_positive)
    print(burden_profiles)
