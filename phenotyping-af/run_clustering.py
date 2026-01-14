"""
Run Time Series Phenotype Clustering

Discovers chronophenotypes by clustering subjects based on their temporal burden patterns.

Usage:
    python run_clustering.py --input_file /path/to/analysis_df.csv --n_clusters 5

Output:
    - Chronophenotypes CSV (burden profiles + cluster labels)
    - Dendrogram visualization
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import DataLoader
from data.preprocessing import ShortBurdenPreprocessor
from clustering.hierarchical import HierarchicalClusterer
from evaluation.metrics import ClusteringMetrics
from evaluation.stability import BootstrapStability
from visualization.plots import PhenotypePlots


def parse_args():
    parser = argparse.ArgumentParser(description='Run time series phenotype clustering')
    
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the cohort analysis CSV file')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters (default: 5)')
    parser.add_argument('--linkage', type=str, default='complete',
                        choices=['ward', 'complete', 'average', 'single'],
                        help='Linkage method (default: complete)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results (default: ./results)')
    parser.add_argument('--label_col', type=str, default='pred',
                        help='Column name for labels (default: pred)')
    parser.add_argument('--patient_col', type=str, default='pat',
                        help='Column name for patient IDs (default: pat)')
    parser.add_argument('--time_col', type=str, default='start_time',
                        help='Column name for event times (default: start_time)')
    parser.add_argument('--search_k', action='store_true',
                        help='Search for optimal number of clusters')
    parser.add_argument('--k_range', type=str, default='2,8',
                        help='Range of k values: min,max (default: 2,8)')
    parser.add_argument('--evaluate_stability', action='store_true',
                        help='Run bootstrap stability analysis')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'
    
    for d in [figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n=== Loading data from {args.input_file} ===")
    loader = DataLoader(label_col=args.label_col, patient_col=args.patient_col)
    df = loader.load_analysis_data(args.input_file, positive_only=False)
    df_positive = df[df[args.label_col] == True] if args.label_col in df.columns else df
    
    n_patients = df[args.patient_col].nunique() if args.patient_col in df.columns else len(df)
    n_patients_positive = df_positive[args.patient_col].nunique() if args.patient_col in df_positive.columns else len(df_positive)
    print(f"Total rows: {len(df)}, Patients: {n_patients}")
    print(f"Positive rows: {len(df_positive)}, Patients with positive events: {n_patients_positive}")
    
    # Compute burden profiles
    print("\n=== Computing burden profiles ===")
    preprocessor = ShortBurdenPreprocessor(
        time_col=args.time_col,
        patient_col=args.patient_col
    )
    burden_profiles = preprocessor.compute_burden_profiles(df, df_positive)
    
    # Initialize metrics and plots
    metrics_calc = ClusteringMetrics()
    plots = PhenotypePlots()
    
    # Search for optimal k if requested
    if args.search_k:
        k_min, k_max = map(int, args.k_range.split(','))
        k_values = list(range(k_min, k_max + 1))
        
        print(f"\n=== Searching optimal k in range {k_values} ===")
        k_metrics = metrics_calc.search_optimal_k(
            burden_profiles.values,
            HierarchicalClusterer,
            k_values,
            linkage=args.linkage
        )
        
        plots.plot_metrics_vs_k(k_metrics, save_path=figures_dir / 'optimal_k_metrics.png')
        
        # Save metrics
        pd.DataFrame(k_metrics).to_csv(tables_dir / 'k_search_metrics.csv', index=False)
    
    # Perform clustering
    print(f"\n=== Clustering with k={args.n_clusters} ===")
    clusterer = HierarchicalClusterer(
        n_clusters=args.n_clusters,
        linkage=args.linkage
    )
    labels = clusterer.fit_predict(burden_profiles.values)
    
    # Print summary
    clusterer.print_summary()
    
    # Compute metrics
    metrics = metrics_calc.compute_all(burden_profiles.values, labels)
    metrics_calc.print_metrics(metrics)
    
    # Create chronophenotypes DataFrame (burden profiles + cluster)
    chronophenotypes = burden_profiles.copy()
    chronophenotypes['cluster'] = labels
    chronophenotypes['mean_burden'] = preprocessor.compute_mean_burden(burden_profiles).values
    
    # Save chronophenotypes
    chronophenotypes.to_csv(tables_dir / f'chronophenotypes_k{args.n_clusters}.csv')
    print(f"\nChronophenotypes saved to {tables_dir / f'chronophenotypes_k{args.n_clusters}.csv'}")
    
    # ================================================================
    # VISUALIZATIONS
    # ================================================================
    print("\n=== Generating visualizations ===")
    
    # 1. Dendrogram
    print("  - Plotting dendrogram...")
    dendrogram_data = clusterer.compute_dendrogram(burden_profiles.values)
    plots.plot_dendrogram(
        dendrogram_data,
        save_path=figures_dir / 'dendrogram.png'
    )
    
    # 2. t-SNE (balanced sampling: max 500 samples per cluster)
    print("  - Plotting t-SNE (balanced)...")
    max_samples_per_cluster = 500
    balanced_idx = chronophenotypes.groupby('cluster', group_keys=False).apply(
        lambda x: x.sample(min(max_samples_per_cluster, len(x)), random_state=42)
    ).index
    balanced_profiles = burden_profiles.loc[balanced_idx]
    balanced_labels = chronophenotypes.loc[balanced_idx, 'cluster'].values
    print(f"    Balanced sample: {len(balanced_idx)} patients (max {max_samples_per_cluster} per cluster)")
    
    plots.plot_tsne(
        balanced_profiles.values,
        balanced_labels,
        save_path=figures_dir / 'tsne.png'
    )
    
    # 3. Combined plot: heatmaps + line plots side by side
    print("  - Plotting combined chronophenotypes (heatmaps + profiles)...")
    plots.plot_chronophenotypes_combined(
        burden_profiles,
        labels,
        max_heatmap_samples=50,
        save_path=figures_dir / 'chronophenotypes_combined.png'
    )
    
    # 4. Clusters heatmap only (for detailed view)
    print("  - Plotting clusters heatmap...")
    max_heatmap_per_cluster = 50
    heatmap_df = chronophenotypes.drop(columns=['mean_burden'], errors='ignore')
    heatmap_df = heatmap_df.groupby('cluster', group_keys=False).apply(
        lambda x: x.sample(min(max_heatmap_per_cluster, len(x)), random_state=42)
    )
    plots.plot_clusters_heatmap(
        heatmap_df,
        cluster_col='cluster',
        save_path=figures_dir / 'clusters_heatmap.png'
    )
    
    # Bootstrap stability if requested
    if args.evaluate_stability:
        print("\n=== Running bootstrap stability analysis ===")
        stability = BootstrapStability(n_bootstraps=100)
        results = stability.evaluate(
            burden_profiles.values,
            labels,
            HierarchicalClusterer,
            args.n_clusters,
            linkage=args.linkage
        )
        stability.print_results(results)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
