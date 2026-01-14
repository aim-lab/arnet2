"""
Clustering evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)


class ClusteringMetrics:
    """
    Computes various clustering quality metrics.
    
    Metrics:
    - Silhouette Score: Measures cluster cohesion vs separation (-1 to 1, higher is better)
    - Davies-Bouldin Index: Ratio of within/between cluster distances (lower is better)
    - Calinski-Harabasz Index: Variance ratio (higher is better)
    
    Example:
        metrics = ClusteringMetrics()
        results = metrics.compute_all(X, labels)
        print(results)
    """
    
    def __init__(self):
        pass
    
    def silhouette(self, X, labels):
        """
        Compute the mean silhouette score.
        
        Args:
            X: Feature matrix.
            labels: Cluster labels.
            
        Returns:
            Mean silhouette score (-1 to 1, higher is better).
        """
        return silhouette_score(X, labels)
    
    def silhouette_per_sample(self, X, labels):
        """
        Compute silhouette score for each sample.
        
        Args:
            X: Feature matrix.
            labels: Cluster labels.
            
        Returns:
            Array of silhouette scores per sample.
        """
        return silhouette_samples(X, labels)
    
    def davies_bouldin(self, X, labels):
        """
        Compute the Davies-Bouldin Index.
        
        Args:
            X: Feature matrix.
            labels: Cluster labels.
            
        Returns:
            Davies-Bouldin score (lower is better).
        """
        return davies_bouldin_score(X, labels)
    
    def calinski_harabasz(self, X, labels):
        """
        Compute the Calinski-Harabasz Index.
        
        Args:
            X: Feature matrix.
            labels: Cluster labels.
            
        Returns:
            Calinski-Harabasz score (higher is better).
        """
        return calinski_harabasz_score(X, labels)
    
    def compute_all(self, X, labels):
        """
        Compute all clustering metrics.
        
        Args:
            X: Feature matrix.
            labels: Cluster labels.
            
        Returns:
            Dictionary with all metric values.
        """
        return {
            'silhouette': self.silhouette(X, labels),
            'davies_bouldin': self.davies_bouldin(X, labels),
            'calinski_harabasz': self.calinski_harabasz(X, labels),
            'n_clusters': len(np.unique(labels)),
            'n_samples': len(labels)
        }
    
    def search_optimal_k(self, X, clusterer_class, k_values, **clusterer_params):
        """
        Search for optimal number of clusters.
        
        Args:
            X: Feature matrix.
            clusterer_class: Clustering class to use (e.g., HierarchicalClusterer).
            k_values: List of k values to try (e.g., [2, 3, 4, 5, 6]).
            **clusterer_params: Additional params for the clusterer.
            
        Returns:
            Dictionary with metrics for each k.
        """
        results = {
            'k': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in k_values:
            clusterer = clusterer_class(n_clusters=k, **clusterer_params)
            labels = clusterer.fit_predict(X)
            
            metrics = self.compute_all(X, labels)
            
            results['k'].append(k)
            results['silhouette'].append(metrics['silhouette'])
            results['davies_bouldin'].append(metrics['davies_bouldin'])
            results['calinski_harabasz'].append(metrics['calinski_harabasz'])
            
            print(f"k={k}: silhouette={metrics['silhouette']:.3f}, "
                  f"davies_bouldin={metrics['davies_bouldin']:.3f}")
        
        return results
    
    def print_metrics(self, metrics):
        """Print metrics in a readable format with interpretation."""
        sil = metrics['silhouette']
        db = metrics['davies_bouldin']
        ch = metrics['calinski_harabasz']
        
        # Interpret silhouette score
        if sil > 0.7:
            sil_interp = "strong structure"
        elif sil > 0.5:
            sil_interp = "reasonable structure"
        elif sil > 0.25:
            sil_interp = "weak structure"
        else:
            sil_interp = "no substantial structure"
        
        # Interpret Davies-Bouldin (lower is better, <1 is generally good)
        if db < 0.5:
            db_interp = "excellent separation"
        elif db < 1.0:
            db_interp = "good separation"
        elif db < 2.0:
            db_interp = "moderate separation"
        else:
            db_interp = "poor separation"
        
        print(f"\n{'='*50}")
        print(f"CLUSTERING METRICS")
        print(f"{'='*50}")
        print(f"\nSilhouette Score: {sil:.3f}")
        print(f"  Range: [-1, +1], higher is better")
        print(f"  Interpretation: {sil_interp}")
        print(f"\nDavies-Bouldin Index: {db:.3f}")
        print(f"  Range: [0, ∞), lower is better")
        print(f"  Interpretation: {db_interp}")
        print(f"\nCalinski-Harabasz Index: {ch:.1f}")
        print(f"  Range: [0, ∞), higher is better")
        print(f"  Note: Scale depends on data, compare across k values")
        print(f"\n{'='*50}")
        print(f"Clusters: {metrics['n_clusters']} | Samples: {metrics['n_samples']}")
        print(f"{'='*50}")

