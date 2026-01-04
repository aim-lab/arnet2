"""
Hierarchical (Agglomerative) Clustering.

This is an exploratory clustering method that produces:
- Cluster labels for input data
- Dendrogram visualization data
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from base import BaseClusterer


class HierarchicalClusterer(BaseClusterer):
    """
    Hierarchical Clustering for discovering phenotypes.
    
    Uses agglomerative (bottom-up) clustering where each sample starts 
    in its own cluster, and clusters are merged as we move up.
    
    Example:
        clusterer = HierarchicalClusterer(n_clusters=5, linkage='complete')
        labels = clusterer.fit_predict(X)
        dendrogram_data = clusterer.compute_dendrogram(X)
    """
    
    def __init__(self, n_clusters=5, linkage='complete', metric='euclidean'):
        """
        Initialize Hierarchical Clusterer.
        
        Args:
            n_clusters: Number of clusters to find.
            linkage: How to measure distance between clusters.
                     - 'complete': maximum distance between clusters (default)
                     - 'ward': minimizes variance within clusters
                     - 'average': average distance between clusters
                     - 'single': minimum distance between clusters
            metric: Distance metric ('euclidean', 'manhattan', etc.).
                    Note: 'ward' only works with 'euclidean'.
        """
        super().__init__(n_clusters=n_clusters)
        
        # ------------------------------------------------------------------ #
        # ---------------------- Set in this child class ------------------- #
        # ------------------------------------------------------------------ #
        
        self.name = "Hierarchical"
        self.linkage = linkage
        self.metric = metric
        
        # ------------------------------------------------------------------ #
        # ---------------------- Internal variables ------------------------ #
        # ------------------------------------------------------------------ #
        
        self.dendrogram_data = None
    
    # ------------------------------------------------------------------ #
    # ------ Override methods from BaseClusterer ----------------------- #
    # ------------------------------------------------------------------ #
    
    def fit_predict(self, X):
        """
        Fit hierarchical clustering and return cluster labels.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features).
            
        Returns:
            labels: Cluster labels for each sample.
        """
        X = np.asarray(X)
        
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            metric=self.metric
        )
        
        self.labels_ = model.fit_predict(X)
        self.is_fitted = True
        
        return self.labels_
    
    def get_visualization_data(self):
        """
        Get dendrogram data for visualization.
        
        Returns:
            Dictionary with dendrogram data, or None if not computed.
        """
        return self.dendrogram_data
    
    # ------------------------------------------------------------------ #
    # ------ Methods specific to Hierarchical Clustering --------------- #
    # ------------------------------------------------------------------ #
    
    def compute_dendrogram(self, X):
        """
        Compute the full dendrogram for the data.
        
        This fits a model with no cluster limit to get the complete 
        hierarchical structure for visualization.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features).
            
        Returns:
            Dictionary with linkage_matrix for plotting.
        """
        X = np.asarray(X)
        
        # Fit with distance_threshold=0 to get full tree
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            linkage=self.linkage,
            metric=self.metric,
            compute_distances=True
        )
        model.fit(X)
        
        # Build linkage matrix for scipy dendrogram
        n_samples = len(model.labels_)
        counts = np.zeros(model.children_.shape[0])
        
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        
        linkage_matrix = np.column_stack([
            model.children_,
            model.distances_,
            counts
        ]).astype(float)
        
        self.dendrogram_data = {
            'linkage_matrix': linkage_matrix,
            'n_samples': n_samples,
            'n_clusters': self.n_clusters
        }
        
        return self.dendrogram_data
    
    def get_labels_for_different_k(self, X, k_values):
        """
        Get cluster labels for multiple values of k.
        
        Useful for exploring different numbers of clusters.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features).
            k_values: List of cluster counts to try (e.g., [2, 3, 4, 5]).
            
        Returns:
            Dictionary mapping k -> cluster labels array.
        """
        X = np.asarray(X)
        results = {}
        
        for k in k_values:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage=self.linkage,
                metric=self.metric
            )
            results[k] = model.fit_predict(X)
        
        return results


if __name__ == '__main__':
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'visualization'))
    
    import numpy as np
    from plots import PhenotypePlots
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 24)  # 100 subjects, 24 time bins
    
    # Cluster the data
    clusterer = HierarchicalClusterer(n_clusters=5, linkage='complete')
    labels = clusterer.fit_predict(X)
    
    # Print summary
    clusterer.print_summary()
    
    # Compute dendrogram
    dendrogram_data = clusterer.compute_dendrogram(X)
    print(f"\nDendrogram computed with {dendrogram_data['n_samples']} samples")
    
    # Plot dendrogram
    plots = PhenotypePlots()
    plots.plot_dendrogram(dendrogram_data, color_threshold=4.2)
