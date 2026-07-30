import os
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


EXCEL_PATH = "./data/experimental_data.xlsx"
RANDOM_STATE = 42
OUT_PNG = "./outputs/model_comparison/xgboost/learning_curves.png"
OUT_CSV = "./outputs/model_comparison/xgboost/r2_results.csv"
BO_EVALS = 60


PROPERTY_NAMES = ['Property_A', 'Property_B', 'Property_C', 'Property_D']

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)



def load_and_prepare(path=EXCEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_excel(path)
    df_num = df.select_dtypes(include=['number']).copy()


    df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_num.dropna(inplace=True)


    if df_num.shape[1] < 11:
        raise ValueError(f"At least 11 numeric columns are required; found {df_num.shape[1]}.")


    X_raw = df_num.iloc[:, :7].copy()


    perf_cols = df_num.columns[-4:]
    y_data = df_num[perf_cols].values


    scaler = StandardScaler().fit(X_raw.values)
    X_scaled = scaler.transform(X_raw.values)

    return X_scaled, y_data, perf_cols.tolist()


def main():
    print(">>> Loading and preprocessing data (7 input features, 4 output properties)...")
    X, y_data, perf_names = load_and_prepare(EXCEL_PATH)
    
    n_properties = y_data.shape[1]
    print(f">>> Property names: {perf_names}")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y_data, test_size=0.2, random_state=RANDOM_STATE
    )

    n_train_total = len(X_train)
    if n_train_total < 18:
        raise ValueError("The training set is too small. Ensure that the spreadsheet contains enough data rows.")

    print(f"\n>>> Data split completed: maximum training size = {n_train_total}, fixed test size = {len(X_test)}")


    space = {
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'n_estimators': hp.quniform('n_estimators', 50, 800, 50),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'gamma': hp.uniform('gamma', 0.0, 5.0),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-3), np.log(10)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-3), np.log(10))
    }

    num_gradients = 8
    chunk_size = n_train_total // num_gradients


    results = {
        "gradient": [],
        "train_size": [],
        "property_name": [],
        "r2_train": [],
        "r2_test": [],
        "best_max_depth": [],
        "best_learning_rate": [],
        "best_n_estimators": []
    }

    print(f"\n>>> Starting XGBoost training across {num_gradients} sample-size gradients with Bayesian optimization for {n_properties} properties...\n")

    for i in range(1, num_gradients + 1):
        current_size = n_train_total if i == num_gradients else i * chunk_size

        X_sub = X_train[:current_size]

        print(f"--- Processing gradient {i}/{num_gradients} (sample size: {current_size}) ---")


        for prop_idx in range(n_properties):
            y_sub = y_train[:current_size, prop_idx]
            y_test_single = y_test[:, prop_idx]
            prop_name = perf_names[prop_idx]

            print(f"  > Optimizing property: {prop_name}")

            def objective(params):
                params['max_depth'] = int(params['max_depth'])
                params['n_estimators'] = int(params['n_estimators'])
                params['min_child_weight'] = int(params['min_child_weight'])

                model = xgb.XGBRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)

                cv_folds = min(5, current_size // 3)
                score = cross_val_score(model, X_sub, y_sub, scoring='r2', cv=cv_folds).mean()
                return {'loss': -score, 'status': STATUS_OK}

            trials = Trials()
            best_hyperparams = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=BO_EVALS,
                trials=trials,
                rstate=np.random.default_rng(RANDOM_STATE),
                show_progressbar=False
            )

            best_params_formatted = {
                'max_depth': int(best_hyperparams['max_depth']),
                'learning_rate': best_hyperparams['learning_rate'],
                'n_estimators': int(best_hyperparams['n_estimators']),
                'subsample': best_hyperparams['subsample'],
                'colsample_bytree': best_hyperparams['colsample_bytree'],
                'gamma': best_hyperparams['gamma'],
                'min_child_weight': int(best_hyperparams['min_child_weight']),
                'reg_alpha': best_hyperparams['reg_alpha'],
                'reg_lambda': best_hyperparams['reg_lambda'],
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }

            final_model = xgb.XGBRegressor(**best_params_formatted)
            final_model.fit(X_sub, y_sub)

            pred_train = final_model.predict(X_sub)
            pred_test = final_model.predict(X_test)

            r2_train = r2_score(y_sub, pred_train)
            r2_test = r2_score(y_test_single, pred_test)

            print(f"    - Best depth: {best_params_formatted['max_depth']}, best tree count: {best_params_formatted['n_estimators']}")
            print(f"    - Training R²: {r2_train:.4f} | Test R²: {r2_test:.4f}")

            results["gradient"].append(i)
            results["train_size"].append(current_size)
            results["property_name"].append(prop_name)
            results["r2_train"].append(r2_train)
            results["r2_test"].append(r2_test)
            results["best_max_depth"].append(best_params_formatted['max_depth'])
            results["best_learning_rate"].append(best_hyperparams['learning_rate'])
            results["best_n_estimators"].append(best_params_formatted['n_estimators'])

        print()

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_CSV, index=False)
    print(f">>> Completed. Detailed XGBoost results and parameter history saved to {OUT_CSV}")




    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['blue', 'orange', 'green', 'red']

    for idx, prop_name in enumerate(perf_names):
        ax = axes[idx]
        mask = df_results['property_name'] == prop_name
        data = df_results[mask]

        ax.plot(data["train_size"], data["r2_train"],
                marker='o', linestyle='-', color=colors[idx], label=f'{prop_name} Train')
        ax.plot(data["train_size"], data["r2_test"],
                marker='s', linestyle='--', color=colors[idx], label=f'{prop_name} Test')

        ax.set_title(f"Learning Curve - {prop_name}")
        ax.set_xlabel("Number of Training Samples")
        ax.set_ylabel("$R^2$ Score")
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.6)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    plt.show()


    plt.figure(figsize=(10, 6))
    for idx, prop_name in enumerate(perf_names):
        mask = df_results['property_name'] == prop_name
        data = df_results[mask]
        plt.plot(data["train_size"], data["r2_test"],
                 marker='s', linestyle='--', color=colors[idx], label=f'{prop_name}')

    plt.title("Test $R^2$ Comparison Across Properties")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("$R^2$ Score")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("./outputs/model_comparison/xgboost/property_comparison.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
