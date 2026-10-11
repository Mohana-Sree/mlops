import pandas as pd

from prefect import flow, task
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


@task(name="Load Dataset")
def load_data():
    """Load the Iris dataset into pandas DataFrames."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    print("Dataset loaded successfully.")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


@task(name="Preprocess Data")
def preprocess_data(X, y):
    """Split the dataset and standardize its features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data preprocessing completed.")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train_scaled, X_test_scaled, y_train, y_test


@task(name="Train ML Model")
def train_model(X_train, y_train):
    """Train a logistic regression classifier."""
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    print("Model training completed.")
    return model


@task(name="Evaluate Model")
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=load_iris().target_names,
            zero_division=0,
        )
    )

    return accuracy


@flow(name="MLOps Iris Training Pipeline", log_prints=True)
def ml_workflow():
    """Orchestrate the end-to-end ML workflow."""
    X, y = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    model = train_model(X_train, y_train)

    accuracy = evaluate_model(model, X_test, y_test)

    print(f"Pipeline completed successfully. Final accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    ml_workflow()
