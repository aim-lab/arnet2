"""
Visualization utilities for time series phenotyping.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from pathlib import Path

# Plot settings
EXT = ".png"
DPI = 400
plt.rcParams.update({'font.size': 26, 'legend.title_fontsize': 20, 'legend.fontsize': 20})


class PhenotypePlots:
    """
    Plotting utilities for time series phenotyping results.
    
    Example:
        plots = PhenotypePlots()
        plots.plot_dendrogram(dendrogram_data, save_path='figures/dendrogram.png')
    """
    
    def __init__(self, figsize=(14, 8), dpi=400):
        """
        Initialize plotter.
        
        Args:
            figsize: Default figure size.
            dpi: Resolution for saving figures.
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_dendrogram(self, dendrogram_data, color_threshold=None, truncate_mode='lastp', 
                        p=30, save_path=None):
        """
        Plot a dendrogram from hierarchical clustering.
        
        Args:
            dendrogram_data: Dictionary with 'linkage_matrix' from compute_dendrogram().
            color_threshold: Height at which to cut for coloring (default: auto).
            truncate_mode: How to truncate the dendrogram ('lastp' shows last p merges).
            p: Number of merges to show when truncate_mode='lastp'.
            save_path: Path to save figure (optional).
            
        Returns:
            Figure object.
        """
        linkage_matrix = dendrogram_data['linkage_matrix']
        
        plt.figure(figsize=(14, 8))
        
        dendrogram(
            linkage_matrix,
            color_threshold=color_threshold,
            truncate_mode=truncate_mode,
            p=p,
            show_leaf_counts=True,
            above_threshold_color='grey',
            leaf_font_size=20
        )
        
        plt.xlabel('Dendrogram')
        plt.ylabel('Height')
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        fig = plt.gcf()
        
        if save_path:
            self.save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_temporal_distribution(self, cluster_df, time_col='time_bin', value_col='burden', 
                                   cluster_col='cluster', patient_col='pat',
                                   cluster_colors=None, save_path=None):
        """
        Plot temporal burden distribution per cluster.
        
        Args:
            cluster_df: DataFrame with time bins, values, and cluster labels.
            time_col: Column name for time bins.
            value_col: Column name for burden values.
            cluster_col: Column name for cluster labels.
            patient_col: Column name for patient identifiers.
            cluster_colors: Dictionary mapping cluster -> color (optional).
            save_path: Path to save figure (optional).
            
        Returns:
            Figure object.
        """
        clusters = sorted(cluster_df[cluster_col].unique())
        n_clusters = len(clusters)
        
        # Default colors
        if cluster_colors is None:
            colors = sns.color_palette('Set1', n_clusters)
            cluster_colors = {c: colors[i] for i, c in enumerate(clusters)}
        
        fig, axs = plt.subplots(nrows=n_clusters, ncols=1, sharey=True, figsize=(10, 6 * n_clusters))
        if n_clusters == 1:
            axs = [axs]
        
        for ax, cluster in zip(axs, clusters):
            cluster_data = cluster_df[cluster_df[cluster_col] == cluster]
            color = cluster_colors.get(cluster, f'C{cluster}')
            
            sns.lineplot(
                data=cluster_data, 
                x=time_col, 
                y=value_col,
                color=color,
                estimator='mean', 
                ax=ax
            )
            
            n_patients = cluster_data[patient_col].nunique() if patient_col in cluster_data.columns else len(cluster_data)
            
            ax.set_ylabel('Burden')
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_ylim([0., 1.])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f'Cluster {cluster}, n={n_patients:,}')
        
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_heatmap(self, data, cluster_labels=None, title="Burden Heatmap", 
                     cmap='vlag', save_path=None):
        """
        Plot heatmap of burden across patients and time.
        
        Args:
            data: Patient x time bin matrix.
            cluster_labels: Optional cluster labels for ordering rows.
            title: Plot title.
            cmap: Colormap to use.
            save_path: Path to save figure (optional).
            
        Returns:
            Figure object.
        """
        plt.figure(figsize=(10, 8))
        
        plot_data = data.copy()
        
        # Sort by cluster if labels provided
        if cluster_labels is not None:
            sort_idx = np.argsort(cluster_labels)
            plot_data = plot_data.iloc[sort_idx]
        
        sns.heatmap(plot_data, cmap=cmap, cbar=True, linewidths=0.5)
        plt.title(title)
        
        plt.tight_layout()
        fig = plt.gcf()
        
        if save_path:
            self.save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_clusters_heatmap(self, heatmap_df, cluster_col='cluster', save_path=None):
        """
        Plot heatmap per cluster (stacked vertically).
        
        Args:
            heatmap_df: DataFrame with cluster column and time bin columns.
            cluster_col: Column name for cluster labels.
            save_path: Path to save figure (optional).
            
        Returns:
            Figure object.
        """
        clusters = sorted(heatmap_df[cluster_col].unique())
        n_clusters = len(clusters)
        
        # Smaller font for heatmaps
        with plt.rc_context({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10}):
            fig, axs = plt.subplots(nrows=n_clusters, sharex=True, figsize=(8, 2 * n_clusters))
            if n_clusters == 1:
                axs = [axs]
            cbar_ax = fig.add_axes([.92, .3, .02, .4])
            
            for i, (ax, cluster) in enumerate(zip(axs, clusters)):
                df_plot = heatmap_df[heatmap_df[cluster_col] == cluster]
                # Get only numeric columns (time bins)
                numeric_cols = [c for c in df_plot.columns if c != cluster_col]
                n_samples = len(df_plot)
                
                sns.heatmap(
                    data=df_plot[numeric_cols], 
                    cmap='vlag', 
                    cbar=(i == 0),
                    linewidths=0, 
                    ax=ax,
                    yticklabels=False,
                    xticklabels=(i == n_clusters - 1),  # Only show x labels on bottom
                    vmin=0., 
                    vmax=1., 
                    cbar_ax=None if i else cbar_ax
                )
                ax.set_ylabel(f'C{cluster} (n={n_samples})', fontsize=10)
                ax.set_xlabel('')
            
            axs[-1].set_xlabel('Hour of Day', fontsize=10)
            axs[-1].set_xticks([0, 6, 12, 18, 24])
            axs[-1].set_xticklabels(['0', '6', '12', '18', '24'])
            
            plt.suptitle('Chronophenotypes Heatmap', fontsize=12)
            plt.tight_layout(rect=[0, 0, .9, 0.96])
        
        if save_path:
            self.save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_chronophenotypes_combined(self, burden_profiles, cluster_labels, 
                                        max_heatmap_samples=50, save_path=None):
        """
        Combined plot: heatmaps on left, mean line plots on right.
        
        Args:
            burden_profiles: DataFrame with patients x time bins.
            cluster_labels: Array of cluster labels.
            max_heatmap_samples: Max samples per cluster for heatmap.
            save_path: Path to save figure (optional).
            
        Returns:
            Figure object.
        """
        clusters = np.unique(cluster_labels)
        n_clusters = len(clusters)
        colors = sns.color_palette('Set1', n_clusters)
        
        # Smaller fonts for this combined plot
        with plt.rc_context({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10}):
            fig, axes = plt.subplots(nrows=n_clusters, ncols=2, 
                                     figsize=(12, 2 * n_clusters),
                                     gridspec_kw={'width_ratios': [1.5, 1]})
            
            if n_clusters == 1:
                axes = axes.reshape(1, 2)
            
            time_bins = list(range(burden_profiles.shape[1]))
            
            for i, cluster in enumerate(clusters):
                mask = cluster_labels == cluster
                cluster_data = burden_profiles.values[mask] if hasattr(burden_profiles, 'values') else burden_profiles[mask]
                n_total = mask.sum()
                
                # Left: Heatmap (sampled)
                ax_heat = axes[i, 0]
                n_show = min(max_heatmap_samples, n_total)
                sample_idx = np.random.RandomState(42).choice(n_total, n_show, replace=False)
                sample_data = cluster_data[sample_idx]
                
                sns.heatmap(
                    sample_data,
                    cmap='vlag',
                    vmin=0, vmax=1,
                    cbar=False,
                    ax=ax_heat,
                    yticklabels=False,
                    xticklabels=(i == n_clusters - 1)
                )
                ax_heat.set_ylabel(f'Cluster {cluster}\n(n={n_total})', fontsize=10)
                if i == n_clusters - 1:
                    ax_heat.set_xlabel('Hour', fontsize=10)
                    ax_heat.set_xticks([0, 6, 12, 18, 24])
                
                # Right: Line plot (mean ± std)
                ax_line = axes[i, 1]
                mean_profile = cluster_data.mean(axis=0)
                std_profile = cluster_data.std(axis=0)
                
                ax_line.plot(time_bins, mean_profile, color=colors[i], linewidth=2)
                ax_line.fill_between(time_bins, mean_profile - std_profile, 
                                     mean_profile + std_profile, color=colors[i], alpha=0.2)
                ax_line.set_ylim([0, 1])
                ax_line.set_xlim([0, 23])
                ax_line.set_ylabel('Burden', fontsize=10)
                ax_line.spines['top'].set_visible(False)
                ax_line.spines['right'].set_visible(False)
                if i == n_clusters - 1:
                    ax_line.set_xlabel('Hour', fontsize=10)
                    ax_line.set_xticks([0, 6, 12, 18, 23])
            
            plt.suptitle('Chronophenotypes: Heatmaps & Mean Profiles', fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            self.save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_metrics_vs_k(self, metrics_dict, highlight_k=None, save_path=None):
        """
        Plot clustering metrics against number of clusters.
        
        Args:
            metrics_dict: Dictionary with 'k' and metric lists from search_optimal_k().
            highlight_k: K value to highlight with a circle (optional).
            save_path: Path to save figure (optional).
            
        Returns:
            Figure object.
        """
        k_values = metrics_dict['k']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_plot = [
            ('silhouette', 'Silhouette Score', 'b'),
            ('davies_bouldin', 'Davies-Bouldin Index', 'orange'),
            ('calinski_harabasz', 'Calinski-Harabasz Index', 'green')
        ]
        
        for ax, (metric, ylabel, color) in zip(axes, metrics_to_plot):
            ax.plot(k_values, metrics_dict[metric], marker='o', linestyle='-', 
                    color=color, markersize=12)
            
            # Highlight specific k if requested
            if highlight_k is not None and highlight_k in k_values:
                idx = k_values.index(highlight_k)
                ax.plot([highlight_k], [metrics_dict[metric][idx]], marker='o', 
                        fillstyle='none', markeredgecolor='r', markersize=24, markeredgewidth=6)
            
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel(ylabel)
            ax.set_xticks(k_values)
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_tsne(self, embeddings, cluster_labels, cluster_colors=None, save_path=None):
        """
        Plot t-SNE visualization of clusters.
        
        Args:
            embeddings: Feature matrix (n_samples, n_features).
            cluster_labels: Cluster labels for each sample.
            cluster_colors: Dictionary mapping cluster -> color (optional).
            save_path: Path to save figure (optional).
            
        Returns:
            Figure object.
        """
        from sklearn.manifold import TSNE
        
        n_clusters = len(np.unique(cluster_labels))
        
        # Default colors
        if cluster_colors is None:
            colors = sns.color_palette('Set1', n_clusters)
            cluster_colors = {c: colors[i] for i, c in enumerate(np.unique(cluster_labels))}
        
        tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=30, 
                    n_iter=2000, init='pca', random_state=42)
        X_embedded = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for label in np.unique(cluster_labels):
            mask = cluster_labels == label
            ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                       c=[cluster_colors[label]], label=f'Cluster {label}')
        
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='right', bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def save_figure(self, fig, path):
        """Save figure to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as PNG
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        print(f"Figure saved to {path}")
        
        # Also save as PDF if it's a PNG
        if str(path).endswith('.png'):
            pdf_path = str(path).replace('.png', '.pdf')
            fig.savefig(pdf_path, dpi=self.dpi, bbox_inches='tight')


if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    # Create sample dendrogram data
    from scipy.cluster.hierarchy import linkage
    
    np.random.seed(42)
    X = np.random.randn(100, 24)
    linkage_matrix = linkage(X, method='complete')
    
    dendrogram_data = {'linkage_matrix': linkage_matrix}
    
    # Plot
    plots = PhenotypePlots()
    plots.plot_dendrogram(dendrogram_data, color_threshold=4.2)
