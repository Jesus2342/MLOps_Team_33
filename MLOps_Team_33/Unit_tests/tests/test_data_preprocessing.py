import pandas as pd
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import select_columns, split_features_and_target, create_column_transformer, apply_encoders

def test_select_columns():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    result = select_columns(df, ['A', 'B'])
    expected = df[['A', 'B']]
    pd.testing.assert_frame_equal(result, expected)

def test_split_features_and_target():
    df = pd.DataFrame({
        'Feature1': [1, 2, 3],
        'Feature2': [4, 5, 6],
        'Target': [0, 1, 0]
    })
    X, y = split_features_and_target(df, 'Target')
    assert X.shape == (3, 2)
    assert y.shape == (3,)

def test_create_column_transformer():
    transformer = create_column_transformer(['Feature1', 'Feature2'])
    assert transformer is not None

def test_apply_encoders():
    df = pd.DataFrame({
        'Feature1': ['A', 'B', 'A'],
        'Feature2': ['X', 'Y', 'X']
    })
    X = df[['Feature1', 'Feature2']]
    y = pd.Series([0, 1, 0])
    column_transformer = create_column_transformer(['Feature1', 'Feature2'])
    X_transformed, y_one_hot = apply_encoders(X, y, column_transformer, LabelEncoder())
    assert X_transformed.shape[1] == 4  # Check number of columns after one-hot encoding
    assert len(y_one_hot) == len(y)  # Check length of y is unchanged