import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning


from bayes_opt import BayesianOptimization


warnings.filterwarnings("ignore", category=ConvergenceWarning)



EXCEL_PATH = "./data/experimental_data.xlsx"
RANDOM_STATE = 42
OUT_PNG = "./outputs/model_comparison/mlp/learning_curves.png"
OUT_CSV = "./outputs/model_comparison/mlp/r2_results.csv"


INIT_POINTS = 5
N_ITER = 60


PROPERTY_NAMES = ['Property_A', 'Property_B', 'Property_C', 'Property_D']

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)



def load_and_prepare(path=EXCEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_excel(path)
    df_num = df.select_dtypes(include=['number']).copy()
    df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_num.dropna(inplace=True)


    X_raw = df_num.iloc[:, :7].copy()
    

    perf_cols = df_num.columns[-4:]
    y_data = df_num[perf_cols].values

    return X_raw.values, y_data, perf_cols.tolist()


def main():
    print(">>> Loading and preprocessing data (7 input features, 4 output properties)...")
    X, y_data, perf_names = load_and_prepare(EXCEL_PATH)
    
    n_properties = y_data.shape[1]
    print(f">>> Property names: {perf_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_data, test_size=0.2, random_state=RANDOM_STATE
    )

    n_train_total = len(X_train)
    print(f"\n>>> Data split completed: maximum training size = {n_train_total}, test size = {len(X_test)}")

    activations = ['relu', 'tanh', 'logistic']
    solvers = ['adam', 'lbfgs']


    pbounds = {
        'num_layers': (1.0, 3.999),
        'l1_size': (16.0, 128.999),
        'l2_size': (16.0, 128.999),
        'l3_size': (16.0, 128.999),
        'act_idx': (0.0, 2.999),
        'solver_idx': (0.0, 1.999),
        'log_alpha': (-5.0, -1.0),
        'log_lr': (-4.0, -1.0)
    }

    num_gradients = 8
    chunk_size = n_train_total // num_gradients
    

    results = {
        "gradient": [],
        "train_size": [],
        "property_name": [],
        "r2_train": [],
        "r2_test": [],
        "best_hidden": [],
        "best_act": [],
        "best_solver": []
    }

    print(f"\n>>> Starting MLP training across {num_gradients} sample-size gradients with Bayesian optimization for {n_properties} properties...\n")

    for i in range(1, num_gradients + 1):
        current_size = n_train_total if i == num_gradients else i * chunk_size
        X_sub = X_train[:current_size]

        print(f"--- Gradient {i}/{num_gradients} (sample size: {current_size}) ---")


        for prop_idx in range(n_properties):
            y_sub = y_train[:current_size, prop_idx]
            y_test_single = y_test[:, prop_idx]
            prop_name = perf_names[prop_idx]

            print(f"  > Optimizing property: {prop_name}")

            def mlp_cv(num_layers, l1_size, l2_size, l3_size, act_idx, solver_idx, log_alpha, log_lr):
                n_layers = int(num_layers)
                h_sizes = [int(l1_size), int(l2_size), int(l3_size)]
                hidden_sizes = tuple(h_sizes[:n_layers])

                act = activations[int(act_idx)]
                sol = solvers[int(solver_idx)]

                model = MLPRegressor(
                    hidden_layer_sizes=hidden_sizes,
                    activation=act,
                    solver=sol,
                    alpha=10 ** log_alpha,
                    learning_rate_init=10 ** log_lr,
                    max_iter=1000,
                    tol=1e-3,
                    early_stopping=(sol == 'adam' and current_size > 50),
                    n_iter_no_change=15,
                    random_state=RANDOM_STATE
                )

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', model)
                ])

                cv_folds = min(5, max(2, current_size // 15))

                try:
                    scores = cross_val_score(pipeline, X_sub, y_sub, cv=cv_folds, scoring='r2')
                    return scores.mean()
                except:
                    return -1.0


            optimizer = BayesianOptimization(f=mlp_cv, pbounds=pbounds, random_state=RANDOM_STATE, verbose=0)
            optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)


            bp = optimizer.max['params']
            best_n = int(bp['num_layers'])
            best_hidden = tuple([int(bp['l1_size']), int(bp['l2_size']), int(bp['l3_size'])][:best_n])
            best_act = activations[int(bp['act_idx'])]
            best_sol = solvers[int(bp['solver_idx'])]


            final_model = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(
                    hidden_layer_sizes=best_hidden,
                    activation=best_act,
                    solver=best_sol,
                    alpha=10 ** bp['log_alpha'],
                    learning_rate_init=10 ** bp['log_lr'],
                    max_iter=4000,
                    tol=1e-3,
                    random_state=RANDOM_STATE
                ))
            ])

            final_model.fit(X_sub, y_sub)

            r2_t = r2_score(y_sub, final_model.predict(X_sub))
            r2_v = r2_score(y_test_single, final_model.predict(X_test))

            print(f"    - Best architecture: {best_hidden}, solver: {best_sol}")
            print(f"    - Training R²: {r2_t:.4f} | Test R²: {r2_v:.4f}")

            results["gradient"].append(i)
            results["train_size"].append(current_size)
            results["property_name"].append(prop_name)
            results["r2_train"].append(r2_t)
            results["r2_test"].append(r2_v)
            results["best_hidden"].append(str(best_hidden))
            results["best_act"].append(best_act)
            results["best_solver"].append(best_sol)

        print()


    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_CSV, index=False)
    print(f"\n>>> Completed. MLP results saved to {OUT_CSV}")




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

    plt.title("MLP Test $R^2$ Comparison Across Properties")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("$R^2$ Score")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("./outputs/model_comparison/mlp/property_comparison.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
