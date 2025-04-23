from typing import Tuple, Dict, Any
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    roc_curve, confusion_matrix
)

from models.model_tuning import tune_model

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MODELS = {
    "RandomForest": RandomForestClassifier,
    "DecisionTree": DecisionTreeClassifier,
    "XGBoost": XGBClassifier,
    "LightGBM": LGBMClassifier,
}


def train_and_evaluate_models(
    X_train, X_test, y_train, y_test
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Tuple], Dict[str, Any]]:
    """
    Train, tune, and evaluate multiple models.

    Returns:
        model_results: metrics + hyperparameters
        models_fitted: trained model objects
        roc_curves: FPR/TPR per model
        confusion_matrices: confusion matrix per model
    """
    model_results = {}
    models_fitted = {}
    roc_curves = {}
    confusion_matrices = {}

    st.write("### 🔍 Hyperparameter Tuning in Progress...")
    for model_name in MODELS:
        with st.spinner(f"Tuning {model_name}..."):
            best_params, _ = tune_model(model_name, X_train, y_train)
            model = MODELS[model_name](**best_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            model_results[model_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "precision": precision,
                "recall": recall,
                "params": best_params
            }

            models_fitted[model_name] = model
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curves[model_name] = (fpr, tpr)
            confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)

    return model_results, models_fitted, roc_curves, confusion_matrices


def display_model_results(model_results: Dict[str, Dict[str, Any]]):
    """Render the performance table and hyperparameters in Streamlit."""
    # --- Performance Table ---
    st.subheader("📊 Model Performance Table")
    table_data = {
        k: {
            "Accuracy": v["accuracy"],
            "F1 Score": v["f1_score"],
            "ROC AUC": v["roc_auc"],
            "Precision": v["precision"],
            "Recall": v["recall"]
        }
        for k, v in model_results.items()
    }
    st.dataframe(pd.DataFrame(table_data).T.style.highlight_max(axis=0))

    # --- Hyperparameters ---
    with st.expander("⚙️ View Tuned Hyperparameters"):
        for model_name, results in model_results.items():
            st.markdown(f"**{model_name}**")
            st.json(results["params"])
