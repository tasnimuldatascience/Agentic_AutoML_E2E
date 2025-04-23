from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Any, Tuple

def preprocess_data(df: Any, target_column: str) -> Tuple[Any, Any, Any, Any]:
    """Preprocess data by encoding target and scaling features."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode target column if it's categorical
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Scale numerical features
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    if numeric_columns.any():
        scaler = MinMaxScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    
    return X, y