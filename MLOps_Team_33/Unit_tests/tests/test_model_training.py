import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from model_training import run_experiments, evaluate_models

def create_dummy_data(num_samples=100, num_features=5, num_classes=3):  # Changed to 3 classes
    """Create dummy classification data for testing."""
    n_informative = 3  # Increased informative features to 3
    n_clusters_per_class = 1  # Reduced to 1 cluster per class

    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_classes=num_classes,
        n_informative=n_informative,
        n_clusters_per_class=n_clusters_per_class,
        random_state=42
    )
    return X, y

def test_run_experiments():
    """Test the run_experiments function with different models."""
    X, y = create_dummy_data()
    
    # Create dummy train-test splits
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Define dummy experiments based on your original notebook
    experiments = [
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=100, max_depth=3),
            (X_train, y_train),
            (X_test, y_test)
        ),
        (
            "XGBoost",
            XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            (X_train, y_train),
            (X_test, y_test)
        ),
        (
            "Logistic Regression",
            LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200),
            (X_train, y_train),
            (X_test, y_test)
        ),
        (
            "K-Nearest Neighbors",
            KNeighborsClassifier(n_neighbors=5),
            (X_train, y_train),
            (X_test, y_test)
        ),
        (
            "MLP",
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam'),
            (X_train, y_train),
            (X_test, y_test)
        ),
        (
            "Support Vector Classifier",
            SVC(kernel='linear', probability=True),
            (X_train, y_train),
            (X_test, y_test)
        )
    ]

    results = run_experiments(experiments)

    # Check if results have the expected model names
    for model_name, _, _, _ in experiments:
        assert model_name in results

def test_evaluate_models():
    """Test the evaluate_models function."""
    X, y = create_dummy_data()
    
    # Create dummy train-test splits
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Define dummy experiments
    experiments = [
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=100, max_depth=3),
            (X_train, y_train),
            (X_test, y_test)
        ),
        (
            "XGBoost",
            XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            (X_train, y_train),
            (X_test, y_test)
        )
    ]

    results = evaluate_models(experiments)

    # Check if results is not empty
    assert len(results) > 0