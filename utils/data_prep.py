import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].median())
    
    # Identify features
    categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal']
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    
    # Standardize numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    return X, y