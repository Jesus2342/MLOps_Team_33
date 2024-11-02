# import mlflow
# import numpy as np
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from mlflow_logging import log_model_metrics, log_experiment_results

# def create_dummy_data(num_samples=100, num_features=5, num_classes=4):
#     """Create dummy classification data for testing."""
#     n_informative = 3  # Adjust this to allow 4 classes with 2 clusters
#     n_clusters_per_class = 1  # Adjust if necessary

#     X, y = make_classification(
#         n_samples=num_samples,
#         n_features=num_features,
#         n_classes=num_classes,
#         n_informative=n_informative,
#         n_clusters_per_class=n_clusters_per_class,
#         random_state=42
#     )
#     return X, y

# def test_log_model_metrics():
#     """Test the log_model_metrics function."""
#     X, y = create_dummy_data()
    
#     # Create dummy train-test splits
#     X_train, X_test = X[:80], X[80:]
#     y_train, y_test = y[:80], y[80:]

#     # Train a dummy model
#     model = RandomForestClassifier(n_estimators=100, max_depth=3)
#     model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Create a dummy report for metrics
#     report = classification_report(y_test, y_pred, output_dict=True)

#     # Start MLflow run
#     with mlflow.start_run(run_name="Test Run"):
#         log_model_metrics(report)

#     # Check if metrics were logged
#     assert mlflow.get_artifact_uri() is not None
#     assert mlflow.active_run() is not None

# def test_log_experiment_results():
#     """Test the log_experiment_results function."""
#     X, y = create_dummy_data()
    
#     # Create dummy train-test splits
#     X_train, X_test = X[:80], X[80:]
#     y_train, y_test = y[:80], y[80:]

#     # Train a dummy model
#     model = RandomForestClassifier(n_estimators=100, max_depth=3)
#     model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Create a dummy report for metrics
#     report = classification_report(y_test, y_pred, output_dict=True)

#     # Prepare experiments and results
#     experiments = [("RandomForest", model)]
#     results_per_model = [report]

#     # Start MLflow run
#     with mlflow.start_run(run_name="Test Experiment Run"):
#         log_experiment_results(experiments, results_per_model)

#     # Check if the metrics were logged
#     assert mlflow.get_artifact_uri() is not None
#     assert mlflow.active_run() is not None

import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from mlflow_logging import log_model_metrics, log_experiment_results

def create_dummy_data(num_samples=100, num_features=5, num_classes=4):
    """Create dummy classification data for testing."""
    n_informative = 3  # Adjust this to allow 4 classes with 2 clusters
    n_clusters_per_class = 1  # Adjust if necessary

    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_classes=num_classes,
        n_informative=n_informative,
        n_clusters_per_class=n_clusters_per_class,
        random_state=42
    )
    return X, y

def test_log_model_metrics():
    """Test the log_model_metrics function."""
    X, y = create_dummy_data()
    
    # Create dummy train-test splits
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Train a dummy model
    model = RandomForestClassifier(n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Create a dummy report for metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    # Set experiment and start MLflow run
    mlflow.set_tracking_uri("http://3.84.228.208:5000")
    mlflow.set_experiment("Test the log_model_metrics function 4")
    with mlflow.start_run(run_name="RandomForestClassifier") as run:
        log_model_metrics(report)

    # Check if metrics were logged
    client = mlflow.tracking.MlflowClient()
    run_id = run.info.run_id  # Retrieve the run ID from the run context
    metrics = client.get_run(run_id).data.metrics
    assert len(metrics) > 0, "Metrics were not logged properly!"

def test_log_experiment_results():
    """Test the log_experiment_results function."""
    X, y = create_dummy_data()
    
    # Create dummy train-test splits
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Train a dummy model
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=3)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

    # Train a dummy XGBoost model
    model_xgb = XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='mlogloss')
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test)
    report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)

    experiments = [
        ("Random Forest", model_rf, {'n_estimators': 100, 'max_depth': 3}),
        ("XGBClassifier", model_xgb, {'n_estimators': 100, 'max_depth': 3})
    ]
    results_per_model = [report_rf, report_xgb]
    
    # Check if the metrics were logged
    metrics = log_experiment_results(experiments, results_per_model)
    assert len(metrics) > 0, "Metrics were not logged properly!"
