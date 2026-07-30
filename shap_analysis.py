import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score


plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.family'] = 'Times New Roman'
warnings.filterwarnings("ignore")


EXCEL_PATH = "./data/experimental_data.xlsx"
WEIGHTS = dict(A=0.35, B=0.15, C=0.20, D=0.30)
RANDOM_STATE = 43
OUT_DIR = "./outputs/shap_analysis/"
N_TRIALS = 10

os.makedirs(OUT_DIR, exist_ok=True)



def score_A(a):
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a, dtype=float);
    m1 = (a > 0) & (a < 5)
    out[m1] = 100.0 * (a[m1] / 5.0);
    m2 = (a >= 5) & (a <= 15);
    out[m2] = 100.0
    m3 = (a > 15);
    out[m3] = 100.0 * np.exp(-(a[m3] - 15.0) / 5.0)
    return out


def score_B(b): return 100.0 * (1.0 - np.exp(-b / 200.0))


def score_C(c): return 100.0 * (1.0 - np.exp(-c / 60.0))


def score_D(d):
    d = np.asarray(d, dtype=float);
    out = np.empty_like(d, dtype=float)
    m1 = d < 3.0;
    m2 = ~m1;
    out[m1] = 60.0 * (d[m1] / 3.0)
    out[m2] = 60.0 + 40.0 * (1.0 - np.exp(-(d[m2] - 3.0) / 8.0))
    return out


def total_score_from_abcd(a, b, c, d):
    return (WEIGHTS['A'] * score_A(a) + WEIGHTS['B'] * score_B(b) +
            WEIGHTS['C'] * score_C(c) + WEIGHTS['D'] * score_D(d))


def load_and_prepare(path=EXCEL_PATH):
    df = pd.read_excel(path)
    df_num = df.select_dtypes(include=['number']).copy()
    df_num.dropna(inplace=True)
    X = df_num.iloc[:, :7].copy()
    a, b, c, d = [df_num.iloc[:, i].to_numpy(float) for i in range(-4, 0)]
    y_score = total_score_from_abcd(a, b, c, d)
    return X, y_score


def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'random_state': RANDOM_STATE, 'n_jobs': -1
    }
    model = xgb.XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X, y, cv=kf, scoring='r2').mean()


def main():

    X, y = load_and_prepare(EXCEL_PATH)


    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS)


    best_model = xgb.XGBRegressor(**study.best_params, n_jobs=-1, random_state=RANDOM_STATE)
    best_model.fit(X, y)


    explainer = shap.TreeExplainer(best_model)
    shap_vals = explainer(X)


    plt.close('all')
    raw_shap = np.abs(shap_vals.values).mean(axis=0)
    norm_shap = raw_shap / np.sum(raw_shap)


    df_imp = pd.DataFrame({'Feature': X.columns, 'Importance': norm_shap}).sort_values('Importance', ascending=True)




    table_save_path = os.path.join(OUT_DIR, "feature_importance.xlsx")

    df_imp_table = df_imp.sort_values('Importance', ascending=False)

    df_imp_table.to_excel(table_save_path, index=False)
    print(f"\n>>> 特征重要性表格已成功保存至: {table_save_path}\n")



    plt.figure(figsize=(12, 8))
    plt.barh(df_imp['Feature'], df_imp['Importance'], color='#008bfb', height=0.6)


    plt.tick_params(axis='y', labelsize=25)
    plt.tick_params(axis='x', labelsize=35)


    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')

    for label in plt.gca().get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_weight('bold')

    plt.xlabel("Normalized Mean |SHAP Value|", fontsize=35, fontweight="bold", fontname="Times New Roman")
    plt.savefig(os.path.join(OUT_DIR, "normalized_feature_importance.png"), dpi=900, bbox_inches='tight')


    plt.close('all')
    shap.summary_plot(shap_vals, X, show=False, plot_size=(12, 8))


    plt.tick_params(axis='y', labelsize=25)
    plt.tick_params(axis='x', labelsize=25)


    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')

    for label in plt.gca().get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_weight('bold')


    plt.xlabel("SHAP Value", fontsize=23, fontweight="bold", fontname="Times New Roman")


    cbar = plt.gcf().axes[-1]
    if cbar:
        cbar.set_ylabel("Feature Value", fontsize=25, fontweight="bold", fontname="Times New Roman")

        cbar.set_yticklabels(cbar.get_yticklabels(), fontsize=23, fontweight="bold", fontname="Times New Roman")
        for label in cbar.get_yticklabels():
            label.set_fontweight("bold")
            label.set_fontname("Times New Roman")

    plt.savefig(os.path.join(OUT_DIR, "shap_beeswarm.png"), dpi=900, bbox_inches='tight')
    print(f">>> 任务完成！所有高分辨率图像和数据表格已保存至 {OUT_DIR}")


if __name__ == "__main__":
    main()
