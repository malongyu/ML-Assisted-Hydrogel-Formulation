import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning

from bayes_opt import BayesianOptimization

warnings.filterwarnings("ignore", category=ConvergenceWarning)


EXCEL_PATH = "./data/experimental_data.xlsx"
RANDOM_STATE = 42
OUT_PNG = "./outputs/model_comparison/gpr/learning_curves.png"
OUT_CSV = "./outputs/model_comparison/gpr/r2_results.csv"

INIT_POINTS = 8
N_ITER = 60


PROPERTY_NAMES = ['Property_A', 'Property_B', 'Property_C', 'Property_D']

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)



def load_and_prepare(path=EXCEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件：{path}")
    df = pd.read_excel(path)
    df_num = df.select_dtypes(include=['number']).copy()
    df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_num.dropna(inplace=True)

    if df_num.shape[1] < 11:
        raise ValueError("数据列数不足 11 列。")

    X_raw = df_num.iloc[:, :7].copy()
    

    perf_cols = df_num.columns[-4:]
    y_data = df_num[perf_cols].values

    scaler = StandardScaler().fit(X_raw.values)
    X_scaled = scaler.transform(X_raw.values)
    return X_scaled, y_data, perf_cols.tolist()


def main():
    print(">>> 正在加载并处理数据 (7 个输入特征, 4 个输出性能)...")
    X, y_data, perf_names = load_and_prepare(EXCEL_PATH)
    
    n_properties = y_data.shape[1]
    print(f">>> 性能指标名称: {perf_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_data, test_size=0.2, random_state=RANDOM_STATE
    )

    n_train_total = len(X_train)
    if n_train_total < 18:
        raise ValueError("训练集样本量太少！")

    print(f"\n>>> 划分完毕：训练集最大 {n_train_total} 个，测试集 {len(X_test)} 个")


    nu_options = [0.5, 1.5, 2.5]




    pbounds = {
        'log_alpha': (-5.0, 1.0),
        'log_length_scale': (-2.0, 2.0),
        'log_constant': (-2.0, 2.0),
        'nu_idx': (0.0, 2.999)
    }

    num_gradients = 8
    chunk_size = n_train_total // num_gradients


    results = {
        "gradient": [],
        "train_size": [],
        "property_name": [],
        "r2_train": [],
        "r2_test": [],
        "best_alpha": [],
        "best_length_scale": [],
        "best_constant": [],
        "best_nu": []
    }

    print(f"\n>>> 开始 {num_gradients} 梯度 GPR 训练，每个梯度对 {n_properties} 个性能指标分别进行贝叶斯优化...\n")

    for i in range(1, num_gradients + 1):
        current_size = n_train_total if i == num_gradients else i * chunk_size
        X_sub = X_train[:current_size]

        print(f"--- 梯度 {i}/{num_gradients} (样本数: {current_size}) ---")


        for prop_idx in range(n_properties):
            y_sub = y_train[:current_size, prop_idx]
            y_test_single = y_test[:, prop_idx]
            prop_name = perf_names[prop_idx]

            print(f"  > 正在优化性能指标: {prop_name}")

            def gpr_cv(log_alpha, log_length_scale, log_constant, nu_idx):
                alpha = 10 ** log_alpha
                length_scale = 10 ** log_length_scale
                constant_value = 10 ** log_constant
                nu = nu_options[int(nu_idx)]


                kernel = C(constant_value) * Matern(length_scale=length_scale, nu=nu)


                model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    optimizer=None,
                    random_state=RANDOM_STATE
                )

                cv_folds = min(5, current_size // 3)
                return cross_val_score(model, X_sub, y_sub, scoring='r2', cv=cv_folds).mean()

            optimizer = BayesianOptimization(
                f=gpr_cv, pbounds=pbounds, random_state=RANDOM_STATE, verbose=0
            )

            optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)

            best_res = optimizer.max['params']
            best_alpha = 10 ** best_res['log_alpha']
            best_length_scale = 10 ** best_res['log_length_scale']
            best_constant = 10 ** best_res['log_constant']
            best_nu = nu_options[int(best_res['nu_idx'])]


            final_kernel = C(best_constant) * Matern(length_scale=best_length_scale, nu=best_nu)
            final_model = GaussianProcessRegressor(
                kernel=final_kernel,
                alpha=best_alpha,
                optimizer=None,
                random_state=RANDOM_STATE
            )
            final_model.fit(X_sub, y_sub)

            r2_train = r2_score(y_sub, final_model.predict(X_sub))
            r2_test = r2_score(y_test_single, final_model.predict(X_test))

            print(
                f"    - 最优 Alpha(噪声): {best_alpha:.4g}, 长度(Length): {best_length_scale:.4g}, 常量(C): {best_constant:.4g}, Nu: {best_nu}")
            print(f"    - 训练集 R²: {r2_train:.4f} | 测试集 R²: {r2_test:.4f}")

            results["gradient"].append(i)
            results["train_size"].append(current_size)
            results["property_name"].append(prop_name)
            results["r2_train"].append(r2_train)
            results["r2_test"].append(r2_test)
            results["best_alpha"].append(best_alpha)
            results["best_length_scale"].append(best_length_scale)
            results["best_constant"].append(best_constant)
            results["best_nu"].append(best_nu)

        print()

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_CSV, index=False)
    print(f">>> 全部跑完！高斯过程(GPR)的结果已保存至 {OUT_CSV}")




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

    plt.title("GPR Test $R^2$ Comparison Across Properties")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("$R^2$ Score")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("./outputs/model_comparison/gpr/property_comparison.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
