"""
Base class for clustering algorithms.

This is a general framework for time series phenotyping.
AF (Atrial Fibrillation) phenotyping is one use case.

To add a new clustering method:
1. Create a new file (e.g., kmeans.py)
2. Create a class that inherits from BaseClusterer
3. Override the required methods marked below
"""

import numpy as np
import pandas as pd


class BaseClusterer:
    """
    Base class for clustering algorithms. All clustering methods inherit from this class.
    
    To create a new clustering method, inherit this class and override the methods
    marked with 'To be overridden in child classes'.
    
    Example usage:
        clusterer = HierarchicalClusterer(n_clusters=5)
        labels = clusterer.fit_predict(X)
    """
    
    def __init__(self, n_clusters=5):
        """
        Initialize the clusterer.
        
        Args:
            n_clusters: Number of clusters to find.
        """
        # ------------------------------------------------------------------ #
        # ---------------------- To be set in child classes ---------------- #
        # ------------------------------------------------------------------ #
        
        self.name = None  # Name of the clustering algorithm (e.g., "hierarchical")
        
        # ------------------------------------------------------------------ #
        # ---------------------- Common variables -------------------------- #
        # ------------------------------------------------------------------ #
        
        self.n_clusters = n_clusters
        self.labels_ = None  # Cluster labels after fitting
        self.is_fitted = False
    
    # ------------------------------------------------------------------ #
    # ------ Methods to be overridden by child classes ----------------- #
    # ------------------------------------------------------------------ #
    
    def fit_predict(self, X):
        """
        Fit the clustering algorithm and return cluster labels.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features).
            
        Returns:
            labels: Cluster labels for each sample.
        """
        raise NotImplementedError("Needs to be implemented by child class.")
    
    def get_visualization_data(self):
        """
        Get data for visualizing the clustering (e.g., dendrogram).
        
        Returns:
            Dictionary with visualization data, or None if not available.
        """
        raise NotImplementedError("Needs to be implemented by child class.")
    
    # ------------------------------------------------------------------ #
    # ------ Common methods (shared by all clustering algorithms) ------ #
    # ------------------------------------------------------------------ #
    
    def fit(self, X):
        """
        Fit the clustering algorithm.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features).
            
        Returns:
            self
        """
        self.labels_ = self.fit_predict(X)
        self.is_fitted = True
        return self
    
    def get_cluster_dataframe(self, subject_ids):
        """
        Create a DataFrame with subject IDs and their cluster assignments.
        
        Args:
            subject_ids: List or array of subject identifiers.
            
        Returns:
            DataFrame with 'patient' and 'cluster' columns.
        """
        if self.labels_ is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        return pd.DataFrame({
            'patient': subject_ids,
            'cluster': self.labels_
        })
    
    def get_cluster_sizes(self):
        """
        Get the number of samples in each cluster.
        
        Returns:
            Dictionary mapping cluster label to count.
        """
        if self.labels_ is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        return dict(zip(unique, counts))
    
    def print_summary(self):
        """Print a summary of the clustering results."""
        if not self.is_fitted:
            print("Clusterer not fitted yet.")
            return
        
        print(f"\n{self.name} Clustering Summary:")
        print(f"  Number of clusters: {self.n_clusters}")
        print(f"  Total samples: {len(self.labels_)}")
        print(f"  Cluster sizes:")
        for cluster, size in self.get_cluster_sizes().items():
            print(f"    Cluster {cluster}: {size} samples")
