# Agentic AutoML

Agentic AutoML is a reflexive, modular AutoML system for binary classification tasks on tabular datasets. It includes:

- **Perception**: Data introspection and analysis
- **Planning**: Dynamic pipeline generation
- **Execution**: Multi-model evaluation
- **Reflection**: Metric-based performance insights
- **Explanation**: SHAP for interpretability + LLM-powered rationale

## Installation
```bash
conda create -n agentic_automl python=3.10 -y
conda activate agentic_automl
pip install -e .
```

## Usage (Streamlit UI)
```bash
streamlit run app.py
```

## License
MIT
