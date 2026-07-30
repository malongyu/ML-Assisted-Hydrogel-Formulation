# Machine Learning-Assisted Hydrogel Formulation for Artificial Vascular Grafts

This workflow was developed to support the data-driven formulation of hydrogel-based artificial vascular grafts. It links material composition and processing parameters to key mechanical properties relevant to vascular graft performance.

This repository contains the machine-learning code used for model comparison, XGBoost-based property prediction, and SHAP analysis. The original computational procedures and hyperparameter settings are retained; only comments, file names, and input/output paths have been standardized.

## Contents

- `benchmarks/`: independent GPR, MLP, random forest, SVR, and XGBoost comparison scripts.
- `xgboost_prediction.py`: full-data XGBoost training and prediction of four mechanical properties.
- `shap_analysis.py`: SHAP-based feature influence analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all commands from the repository root:

```bash
python benchmarks/gpr.py
python benchmarks/mlp.py
python benchmarks/random_forest.py
python benchmarks/svr.py
python benchmarks/xgboost.py
python xgboost_prediction.py
python shap_analysis.py
```

Generated models, tables, and figures are saved under `outputs/`. XGBoost prediction uses CUDA by default.
