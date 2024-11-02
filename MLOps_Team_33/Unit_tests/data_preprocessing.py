import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def select_columns(df, columns_of_interest):
    """
    Selects specified columns from the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_of_interest (list): List of columns to select.
    
    Returns:
        pd.DataFrame: DataFrame with selected columns.
    """
    return df[columns_of_interest]

def split_features_and_target(df, target_column):
    """
    Splits DataFrame into features and target.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
    
    Returns:
        tuple: Features and target as separate objects.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def create_column_transformer(columns_to_encode):
    """
    Creates a ColumnTransformer for One-Hot Encoding.
    
    Args:
        columns_to_encode (list): List of columns to one-hot encode.
    
    Returns:
        ColumnTransformer: Configured ColumnTransformer object.
    """
    enc = OneHotEncoder(sparse_output=False)
    ct = ColumnTransformer(transformers=[("OneHot", enc, columns_to_encode)])
    return ct

def apply_encoders(X, y, column_transformer, label_encoder):
    """
    Applies one-hot encoding and label encoding to features and target.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        column_transformer (ColumnTransformer): ColumnTransformer for encoding.
        label_encoder (LabelEncoder): LabelEncoder for target.
    
    Returns:
        tuple: Transformed features and one-hot encoded target.
    """
    X_transformed = column_transformer.fit_transform(X)
    y_one_hot = OneHotEncoder(sparse_output=False).fit_transform(y.values.reshape(-1, 1))
    y_encoded = label_encoder.fit_transform(y)
    return X_transformed, y_one_hot