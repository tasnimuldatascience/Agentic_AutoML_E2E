import streamlit as st
import pandas as pd
from typing import Tuple, Any
import openai
from sklearn.model_selection import train_test_split

from utils.preprocessing import preprocess_data
from utils.visualization import plot_label_distribution_and_pca, plot_roc_curves, plot_confusion_matrices
from models.model_training import train_and_evaluate_models, display_model_results
from models.llm_explanation import explain_with_llm

from config import Config

# Set OpenAI API Key
openai.api_key = Config.get_openai_api_key()

def display_header() -> None:
    """Displays the header of the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("🤖 Agentic AutoML: Model Comparison, Explanation & SHAP Visualizations")

def upload_file() -> pd.DataFrame:
    """Uploads the CSV file and returns a pandas DataFrame."""
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.DataFrame()

def main() -> None:
    """Main function to run the Streamlit app."""
    display_header()

    # File upload
    df = upload_file()
    if df.empty:
        return

    st.write(df.head())

    # Select target column for prediction
    target_column = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)

    # Preprocess the data
    X, y = preprocess_data(df, target_column)
    plot_label_distribution_and_pca(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Train models
    model_results, models_fitted, roc_curves, confusion_matrices = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    display_model_results(model_results)
    
    plot_roc_curves(roc_curves)
    plot_confusion_matrices(confusion_matrices)

    best_model_name = max(model_results, key=lambda x: model_results[x]["accuracy"])
    best_model = models_fitted[best_model_name]
     # --- Call explain_with_llm ---
    llm_explanation = explain_with_llm(
        results=model_results,
        shap_values=None,  # will be computed inside explain_with_llm
        best_model_name=best_model_name,
        model=best_model,
        X_train=X_train
    )
    
    st.write(llm_explanation)

if __name__ == "__main__":
    main()
