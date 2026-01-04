"""
Bootstrap stability analysis for clustering.
"""

from collections import defaultdict
import numpy as np
from sklearn.metrics import adjusted_rand_score


class BootstrapStability:
    """
    Evaluates clustering stability using bootstrap resampling.
    
    Measures how consistently samples are assigned to the same cluster
    across different bootstrap samples.
    
    Example:
        stability = BootstrapStability(n_bootstraps=100)
        results = stability.evaluate(X, labels, HierarchicalClusterer, n_clusters=5)
        stability.print_results(results)
    """
    
    def __init__(self, n_bootstraps=100, sample_fraction=0.7, random_state=42):
        """
        Initialize bootstrap stability analyzer.
        
        Args:
            n_bootstraps: Number of bootstrap iterations.
            sample_fraction: Fraction of data to sample each time.
            random_state: Random seed for reproducibility.
        """
        self.n_bootstraps = n_bootstraps
        self.sample_fraction = sample_fraction
        self.random_state = random_state
    
    def evaluate(self, X, base_labels, clusterer_class, n_clusters, **clusterer_params):
        """
        Evaluate clustering stability.
        
        Args:
            X: Feature matrix.
            base_labels: Cluster labels from baseline clustering.
            clusterer_class: Clustering class to use.
            n_clusters: Number of clusters.
            **clusterer_params: Additional params for clusterer.
            
        Returns:
            Dictionary with stability metrics.
        """
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X)
        base_labels = np.asarray(base_labels)
        
        ari_values = []
        cluster_jaccard = defaultdict(list)
        
        # Get indices for each cluster
        cluster_indices = {}
        for c in range(n_clusters):
            idx = np.where(base_labels == c)[0]
            if len(idx) > 0:
                cluster_indices[c] = idx
        
        cluster_sizes = {c: len(idx) for c, idx in cluster_indices.items()}
        
        print(f"Running {self.n_bootstraps} bootstrap iterations...")
        
        for i in range(self.n_bootstraps):
            # Stratified sampling from each cluster
            sampled_idx = []
            for c, idx in cluster_indices.items():
                n_sample = max(1, int(self.sample_fraction * len(idx)))
                sampled_idx.append(rng.choice(idx, size=n_sample, replace=False))
            sampled_idx = np.concatenate(sampled_idx)
            
            X_sub = X[sampled_idx]
            base_sub = base_labels[sampled_idx]
            
            # Re-cluster
            clusterer = clusterer_class(n_clusters=n_clusters, **clusterer_params)
            new_labels = clusterer.fit_predict(X_sub)
            
            # Compute Adjusted Rand Index
            ari = adjusted_rand_score(base_sub, new_labels)
            ari_values.append(ari)
            
            # Compute per-cluster Jaccard
            for c in range(n_clusters):
                orig_idx = np.where(base_sub == c)[0]
                if len(orig_idx) == 0:
                    continue
                
                best_jaccard = 0.0
                for new_c in range(n_clusters):
                    new_idx = np.where(new_labels == new_c)[0]
                    if len(new_idx) == 0:
                        continue
                    
                    inter = np.intersect1d(orig_idx, new_idx).size
                    union = np.union1d(orig_idx, new_idx).size
                    jaccard = inter / union if union > 0 else 0.0
                    best_jaccard = max(best_jaccard, jaccard)
                
                cluster_jaccard[c].append(best_jaccard)
        
        results = {
            'ari_values': np.array(ari_values),
            'ari_mean': np.mean(ari_values),
            'ari_std': np.std(ari_values),
            'cluster_sizes': cluster_sizes,
            'cluster_jaccard': {c: np.array(vals) for c, vals in cluster_jaccard.items()}
        }
        
        return results
    
    def print_results(self, results):
        """Print a summary of stability results."""
        ari = results['ari_values']
        
        print(f"\n=== Clustering Stability Results ===")
        print(f"\nGlobal Stability (Adjusted Rand Index):")
        print(f"  Mean: {results['ari_mean']:.3f} ± {results['ari_std']:.3f}")
        print(f"  Range: [{ari.min():.3f}, {ari.max():.3f}]")
        
        print(f"\nPer-Cluster Stability (Jaccard Index):")
        for c, size in results['cluster_sizes'].items():
            if c in results['cluster_jaccard']:
                vals = results['cluster_jaccard'][c]
                print(f"  Cluster {c} (n={size}): {vals.mean():.3f} ± {vals.std():.3f}")

