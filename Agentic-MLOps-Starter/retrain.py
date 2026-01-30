import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_feedback_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feedback_dir = os.path.join(base_dir, "data", "feedback")

    dfs = []
    for file in os.listdir(feedback_dir):
        if file.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(feedback_dir, file)))

    return pd.concat(dfs, ignore_index=True)

def main():
    mlflow.set_experiment("agentic-mlops-retraining")

    df = load_feedback_data()

    X = df[["agent_id", "tool_name", "status"]]
    y = df["trade_value"]

    cat_features = ["agent_id", "tool_name", "status"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression())
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

        print(f"Training complete. MSE: {mse}")

if __name__ == "__main__":
    main()
