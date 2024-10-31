from sklearn.metrics import accuracy_score, classification_report

def run_experiments(experiments):
    """
    Runs a series of classification experiments.
    
    Args:
        experiments (list): List of tuples containing experiment details.
    
    Returns:
        dict: A dictionary with model names and their corresponding accuracy scores.
    """
    results = {}
    
    for name, model, (X_train, y_train), (X_test, y_test) in experiments:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = classification_report(y_test, y_pred, output_dict=True)
        
    return results

def evaluate_models(experiments):
    """
    Evaluates multiple classification models and generates classification reports.
    
    Args:
        experiments (list): List of tuples containing model details.
    
    Returns:
        list: A list of classification reports for each model.
    """
    results = []
    
    for model_name, model, train_set, test_set in experiments:
        X_train, y_train = train_set
        X_test, y_test = test_set
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results.append(report)
    
    return results
