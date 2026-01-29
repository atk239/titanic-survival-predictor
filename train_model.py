"""
Train a Logistic Regression model on the Titanic dataset.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns


def load_and_preprocess_data():
    """Load Titanic dataset and preprocess it."""
    # Load dataset from seaborn
    df = sns.load_dataset('titanic')

    # Create working copy
    data = df.copy()

    # Handle missing values for base features
    data['age'] = data['age'].fillna(data['age'].median())
    data['fare'] = data['fare'].fillna(data['fare'].median())

    # Encode sex (male=0, female=1)
    data['sex_encoded'] = data['sex'].map({'male': 0, 'female': 1})

    # Create marital status based on sibsp (spouse/siblings aboard)
    # 0 = single/bachelor, 1 = married, 2 = widowed (estimated)
    np.random.seed(42)
    data['marital_status'] = data.apply(lambda row:
        1 if row['sibsp'] > 0 else np.random.choice([0, 2], p=[0.85, 0.15]), axis=1)

    # Generate synthetic height based on sex and age
    # Males: mean ~170cm, Females: mean ~160cm, with some variation by age
    data['height'] = data.apply(lambda row:
        np.random.normal(170, 8) if row['sex'] == 'male' else np.random.normal(160, 7),
        axis=1)
    # Adjust for children
    data.loc[data['age'] < 18, 'height'] = data.loc[data['age'] < 18].apply(
        lambda row: 100 + row['age'] * 3.5 + np.random.normal(0, 5), axis=1)

    # Select features
    features = ['pclass', 'sex_encoded', 'age', 'fare', 'marital_status', 'height']
    target = 'survived'

    X = data[features].copy()
    X.columns = ['pclass', 'sex', 'age', 'fare', 'marital_status', 'height']
    y = data[target]

    return X, y


def train_model():
    """Train and save the model."""
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    # Evaluate on training data
    train_accuracy = model.score(X_scaled, y)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Save model and scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model saved to model.pkl")
    print("Scaler saved to scaler.pkl")

    return model, scaler


if __name__ == '__main__':
    train_model()
