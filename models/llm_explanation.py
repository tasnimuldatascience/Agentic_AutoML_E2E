import shap
import numpy as np
import pandas as pd
import openai
import streamlit as st
import matplotlib.pyplot as plt
from typing import Any, Dict


def explain_with_llm(
    results: Dict[str, Dict[str, float]],
    shap_values: Any,
    best_model_name: str,
    model: Any,
    X_train: pd.DataFrame
) -> str:
    try:
        # --- Progress Bar Initialization ---
        st.write("🔍 Computing SHAP values. Please wait...")
        progress_bar = st.progress(0)

        # Step 1: Initialize Explainer
        progress_bar.progress(20)
        explainer = shap.Explainer(model, X_train)

        # Step 2: Compute SHAP values
        progress_bar.progress(60)
        shap_values = explainer(X_train)

        # Step 3: Compute feature importance
        progress_bar.progress(80)
        feature_names = X_train.columns
        feature_importance = np.mean(np.abs(shap_values.values), axis=0)

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        top_features = feature_importance_df.head(5)

        # Step 4: Visualization with Beeswarm
        progress_bar.progress(90)
        st.write(f"### 🐝 SHAP Beeswarm Plot for {best_model_name}")
        plt.clf()
        fig_shap = plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=10, show=False)
        st.pyplot(fig_shap)

        # Finalize progress
        progress_bar.progress(100)


        # --- Construct LLM Prompt ---
        prompt = f"""
        These are the performances of the models in a binary classification task:

        {results}

        The model with the highest accuracy is **{best_model_name}**. Here are its metrics:

        - Accuracy: {results[best_model_name]['accuracy']}
        - F1 Score: {results[best_model_name]['f1_score']}
        - Precision: {results[best_model_name]['precision']}
        - Recall: {results[best_model_name]['recall']}

        Additionally, SHAP (Shapley Additive Explanations) values help us understand feature influence.
        The top 5 most important features are:

        {top_features.to_string(index=False)}

        Please explain:
        1. Why {best_model_name} might be the best model based on the performance metrics.
        2. How the SHAP values reflect the model's behavior and which features are the most influential.
        3. For each of the top 5 features, explain the direction of influence (positive or negative) on predictions.
        """

        st.subheader("🤖 Generating LLM-based Explanation...")

        # --- Call OpenAI API ---
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with expertise in machine learning and model interpretability."},
                {"role": "user", "content": prompt}
            ]
        )

        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        st.warning(f"LLM explanation failed: {e}")
        return f"❌ LLM Error: {e}"
