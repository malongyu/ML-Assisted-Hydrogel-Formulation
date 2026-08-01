# Machine Learning-Assisted Hydrogel Formulation for Artificial Vascular Grafts

This workflow was developed to support the data-driven formulation of hydrogel-based artificial vascular grafts. It links material composition and processing parameters to key mechanical properties relevant to vascular graft performance.

## Contents

- `model/`: independent GPR, MLP, random forest, SVR, and XGBoost comparison scripts.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all commands from the repository root:

```bash
python model/gpr.py
python model/mlp.py
python model/random_forest.py
python model/svr.py
python model/xgboost.py
```
