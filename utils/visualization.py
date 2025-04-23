import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.decomposition import PCA

def plot_label_distribution_and_pca(X, y):
    """
    Display label distribution and PCA plot side by side in Streamlit.

    Parameters:
    - X: pd.DataFrame or np.ndarray — Feature matrix
    - y: pd.Series or np.ndarray — Target labels
    """
    st.write("### Label Distribution and PCA Plot (Side by Side)")

    # --- Label Distribution Bar Chart ---
    label_distribution = y.value_counts()
    fig_label_dist, ax_label_dist = plt.subplots(figsize=(5, 3))
    label_distribution.plot(kind='bar', ax=ax_label_dist, color='skyblue')
    ax_label_dist.set_ylabel('Count')
    ax_label_dist.set_xlabel('Label')
    ax_label_dist.set_title('Label Distribution (Target Column)')

    # --- PCA Plot ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig_pca, ax_pca = plt.subplots(figsize=(5, 3))
    scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=10)
    ax_pca.set_xlabel('PCA 1')
    ax_pca.set_ylabel('PCA 2')
    ax_pca.set_title('PCA Plot')
    
    # Add color legend to PCA plot
    legend1 = ax_pca.legend(*scatter.legend_elements(), title="Classes")
    ax_pca.add_artist(legend1)

    # --- Display in Streamlit ---
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_label_dist)
    with col2:
        st.pyplot(fig_pca)


def plot_roc_curves(roc_curves: dict):
    """
    Plots ROC curves for all models.
    
    Args:
        roc_curves (dict): Dictionary mapping model names to (fpr, tpr) tuples.
    """
    st.subheader("📈 ROC Curve Comparison")
    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))  # Compact display
    for name, (fpr, tpr) in roc_curves.items():
        ax_roc.plot(fpr, tpr, label=name)
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)


def plot_confusion_matrices(confusion_matrices: dict):
    """
    Plots confusion matrices in a 2x2 grid layout.

    Args:
        confusion_matrices (dict): Dictionary mapping model names to confusion matrix arrays.
    """
    st.subheader("📊 Confusion Matrices for each Model")

    num_models = len(confusion_matrices)
    num_plots = min(num_models, 4)  # Just 4 for the 2x2 grid

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, (model_name, cm) in enumerate(confusion_matrices.items()):
        if i >= 4:  # only handle first 4 models
            break

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=axes[i]
        )
        axes[i].set_title(f"{model_name}")
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Hide unused subplots if fewer than 4
    for j in range(i+1, 4):
        fig.delaxes(axes[j])

    fig.tight_layout()
    st.pyplot(fig)