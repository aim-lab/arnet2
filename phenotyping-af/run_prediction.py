"""
Run Phenotype Prediction

Train or use a predictor model to classify new subjects into chronophenotypes.

Two modes:
1. train: Train a predictor on burden profiles + cluster labels, save the model
2. predict: Load a saved model and predict chronophenotypes for new data

Usage:
    # Train
    python run_prediction.py train --chronophenotypes_file chronophenotypes.csv --model_output model.pkl
    
    # Predict
    python run_prediction.py predict --model_file model.pkl --input_file new_burden_profiles.csv --output_file predictions.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import DataLoader
from data.preprocessing import ShortBurdenPreprocessor
from prediction.predictor import PhenotypePredictor

# Chronophenotype names from paper
CHRONOPHENOTYPE_NAMES = {
    0: 'Nocturnal-to-Morning',
    1: 'Evening-to-Early Morning',
    2: 'Daytime',
    3: 'Persistent AF',
    4: 'Non-AF'
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train or run phenotype prediction')
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train a new predictor')
    train_parser.add_argument('--chronophenotypes_file', type=str, required=True,
                              help='Path to chronophenotypes CSV (burden profiles + cluster column)')
    train_parser.add_argument('--model_output', type=str, default='./models/predictor.pkl',
                              help='Path to save model')
    train_parser.add_argument('--n_neighbors', type=int, default=5,
                              help='Number of neighbors for KNN (default: 5)')
    
    # Predict mode
    predict_parser = subparsers.add_parser('predict', help='Predict chronophenotypes')
    predict_parser.add_argument('--model_file', type=str, required=True,
                                help='Path to saved model')
    predict_parser.add_argument('--input_file', type=str, required=True,
                                help='Path to input burden profiles CSV')
    predict_parser.add_argument('--output_file', type=str, default='./results/predictions.csv',
                                help='Path to save predictions')
    
    return parser.parse_args()


def train_mode(args):
    """Train a new phenotype predictor."""
    print("\n=== Training Phenotype Predictor ===")
    
    # Load chronophenotypes (burden profiles + cluster labels)
    print(f"\nLoading chronophenotypes from {args.chronophenotypes_file}...")
    df = pd.read_csv(args.chronophenotypes_file, index_col=0)
    print(f"  Shape: {df.shape}")
    
    # Separate burden profiles from cluster labels
    if 'cluster' not in df.columns:
        raise ValueError("Chronophenotypes file must have a 'cluster' column")
    
    cluster_labels = df['cluster'].values
    
    # Get burden profile columns (numeric, excluding cluster and mean_burden)
    profile_cols = [c for c in df.columns if c not in ['cluster', 'mean_burden']]
    burden_profiles = df[profile_cols].values
    
    print(f"  Subjects: {len(df)}")
    print(f"  Burden profile dimensions: {burden_profiles.shape[1]}")
    
    # Train
    predictor = PhenotypePredictor(n_neighbors=args.n_neighbors)
    predictor.fit(burden_profiles, cluster_labels)
    
    # Save
    predictor.save(args.model_output)
    
    # Summary
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nTraining Summary:")
    print(f"  Total samples: {len(cluster_labels)}")
    print(f"  Chronophenotypes: {len(unique)}")
    for label, count in zip(unique, counts):
        name = CHRONOPHENOTYPE_NAMES.get(label, f'Cluster {label}')
        print(f"    {name}: {count} subjects")


def predict_mode(args):
    """Predict chronophenotypes using saved model."""
    print("\n=== Chronophenotype Prediction ===")
    
    # Load model
    predictor = PhenotypePredictor()
    predictor.load(args.model_file)
    
    # Load input burden profiles
    print(f"\nLoading burden profiles from {args.input_file}...")
    burden_profiles = pd.read_csv(args.input_file, index_col=0)
    print(f"  Samples: {len(burden_profiles)}")
    
    # Predict
    print("\nPredicting chronophenotypes...")
    predictions = predictor.predict_with_confidence(burden_profiles.values)
    predictions.insert(0, 'subject', burden_profiles.index)
    
    # Rename column for clarity
    predictions = predictions.rename(columns={'phenotype': 'chronophenotype'})
    
    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Summary
    print(f"\nPrediction Summary:")
    for phenotype in sorted(predictions['chronophenotype'].unique()):
        mask = predictions['chronophenotype'] == phenotype
        count = mask.sum()
        conf = predictions.loc[mask, 'confidence'].mean()
        name = CHRONOPHENOTYPE_NAMES.get(phenotype, f'Cluster {phenotype}')
        print(f"  {name}: {count} subjects (avg confidence: {conf:.3f})")


def main():
    args = parse_args()
    
    if args.mode is None:
        print("Error: Please specify mode (train or predict)")
        print("Use --help for usage information")
        return
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)


if __name__ == '__main__':
    main()
