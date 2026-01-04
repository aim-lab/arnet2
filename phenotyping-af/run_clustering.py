"""
Run Time Series Phenotype Clustering

Discovers chronophenotypes by clustering subjects based on their temporal burden patterns.

Usage:
    python run_clustering.py --input_file /path/to/analysis_df.csv --n_clusters 5

Output:
    - Chronophenotypes CSV (burden profiles + cluster labels)
    - Dendrogram visualization
"""

import sys
import os

# Add paths so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'clustering'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'visualization'))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from loader import DataLoader
from preprocessing import ShortBurdenPreprocessor
from hierarchical import HierarchicalClusterer
from metrics import ClusteringMetrics
from stability import BootstrapStability
from plots import PhenotypePlots


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
    
    print(f"Total records: {len(df)}, Positive records: {len(df_positive)}")
    
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
    
    # Compute and plot dendrogram
    print("\n=== Computing dendrogram ===")
    dendrogram_data = clusterer.compute_dendrogram(burden_profiles.values)
    plots.plot_dendrogram(
        dendrogram_data,
        save_path=figures_dir / 'dendrogram.png'
    )
    
    # Create chronophenotypes DataFrame (burden profiles + cluster)
    chronophenotypes = burden_profiles.copy()
    chronophenotypes['cluster'] = labels
    chronophenotypes['mean_burden'] = preprocessor.compute_mean_burden(burden_profiles).values
    
    # Save chronophenotypes
    chronophenotypes.to_csv(tables_dir / f'chronophenotypes_k{args.n_clusters}.csv')
    print(f"\nChronophenotypes saved to {tables_dir / f'chronophenotypes_k{args.n_clusters}.csv'}")
    
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
