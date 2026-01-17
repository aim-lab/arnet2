"""
Phenotype Predictor - Predicts chronophenotype labels on new data.

After clustering, use this class to:
1. Train a classifier on the cluster labels
2. Save the model
3. Load and predict on new subjects
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier


class PhenotypePredictor:
    """
    Predicts chronophenotype labels for new subjects.
    
    This trains a classifier (KNN by default) on burden profiles and their
    cluster labels, then can predict chronophenotypes for new subjects.
    
    Example:
        # Train and save
        predictor = PhenotypePredictor(n_neighbors=5)
        predictor.fit(burden_profiles, cluster_labels)
        predictor.save('model.pkl')
        
        # Load and predict
        predictor = PhenotypePredictor()
        predictor.load('model.pkl')
        predictions = predictor.predict(new_burden_profiles)
    """
    
    def __init__(self, n_neighbors=5):
        """
        Initialize the phenotype predictor.
        
        Args:
            n_neighbors: Number of neighbors for KNN classifier.
        """
        self.n_neighbors = n_neighbors
        self.model = None
        self.n_clusters = None
        self.is_fitted = False
    
    def fit(self, X, labels):
        """
        Train the predictor on burden profiles and cluster labels.
        
        Args:
            X: Burden profiles matrix, shape (n_samples, n_time_bins).
            labels: Cluster labels from the clustering step.
            
        Returns:
            self
        """
        X = np.asarray(X)
        labels = np.asarray(labels)
        
        if X.shape[0] != len(labels):
            raise ValueError(f"X has {X.shape[0]} samples but labels has {len(labels)}")
        
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(X, labels)
        
        self.n_clusters = len(np.unique(labels))
        self.is_fitted = True
        
        print(f"Predictor trained on {len(labels)} samples, {self.n_clusters} chronophenotypes")
        
        return self
    
    def predict(self, X):
        """
        Predict chronophenotype labels for new data.
        
        Args:
            X: Burden profiles matrix, shape (n_samples, n_time_bins).
            
        Returns:
            Predicted chronophenotype labels.
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")
        
        return self.model.predict(np.asarray(X))
    
    def predict_proba(self, X):
        """
        Predict probability of each chronophenotype.
        
        Args:
            X: Burden profiles matrix, shape (n_samples, n_time_bins).
            
        Returns:
            Array of shape (n_samples, n_chronophenotypes) with probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")
        
        return self.model.predict_proba(np.asarray(X))
    
    def predict_with_confidence(self, X):
        """
        Predict chronophenotype labels with confidence scores.
        
        Args:
            X: Burden profiles matrix, shape (n_samples, n_time_bins).
            
        Returns:
            DataFrame with 'phenotype' and 'confidence' columns.
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        confidence = probabilities.max(axis=1)
        
        return pd.DataFrame({
            'phenotype': predictions,
            'confidence': confidence
        })
    
    def save(self, filepath):
        """
        Save the trained predictor to disk.
        
        Args:
            filepath: Path to save the model (e.g., 'model.pkl').
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'model': self.model,
            'n_neighbors': self.n_neighbors,
            'n_clusters': self.n_clusters,
        }
        
        joblib.dump(save_data, filepath)
        print(f"Predictor saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a trained predictor from disk.
        
        Args:
            filepath: Path to the saved model.
            
        Returns:
            self
        """
        data = joblib.load(filepath)
        
        self.model = data['model']
        self.n_neighbors = data['n_neighbors']
        self.n_clusters = data['n_clusters']
        self.is_fitted = True
        
        print(f"Predictor loaded from {filepath}")
        
        return self


if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 24)
    labels = np.random.randint(0, 5, 100)
    
    # Train predictor
    predictor = PhenotypePredictor(n_neighbors=5)
    predictor.fit(X_train, labels)
    
    # Predict on new data
    X_new = np.random.randn(10, 24)
    predictions = predictor.predict_with_confidence(X_new)
    print(predictions)
