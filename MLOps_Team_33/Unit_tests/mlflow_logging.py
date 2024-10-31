import mlflow
import mlflow.sklearn
import mlflow.xgboost

def log_model_metrics(report):
    """
    Logs classification metrics to MLflow.
    
    Args:
        report (dict): Classification report as a dictionary.
    
    Returns:
        None
    """
    for class_label in range(len(report)):
        mlflow.log_metric(f'accuracy_class_{class_label}', report[str(class_label)]['precision'])
        mlflow.log_metric(f'recall_class_{class_label}', report[str(class_label)]['recall'])
        mlflow.log_metric(f'f1_class_{class_label}', report[str(class_label)]['f1-score'])

def log_experiment_results(experiments, results_per_model):
    """
    Logs model parameters and metrics to MLflow for each experiment.
    
    Args:
        experiments (list): List of tuples containing model details.
        results_per_model (list): List of classification reports for each model.
    
    Returns:
        None
    """
    mlflow.set_experiment("Student Performance Analysis Model")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    for i, (model_name, model) in enumerate(experiments):
        report = results_per_model[i]
        
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            
            for class_label in range(len(report)):
                mlflow.log_metric(f'accuracy_class_{class_label}', report[str(class_label)]['precision'])
                mlflow.log_metric(f'recall_class_{class_label}', report[str(class_label)]['recall'])
                mlflow.log_metric(f'f1_class_{class_label}', report[str(class_label)]['f1-score'])
            
            if "XGB" in model_name:
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
