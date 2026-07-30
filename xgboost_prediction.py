import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
warnings.filterwarnings('ignore', category=UserWarning)


TRAIN_PATH = "./data/experimental_data.xlsx"
PREDICT_PATH = "./data/candidate_formulations.xlsx"
RANDOM_STATE = 43
OUT_DIR = "./outputs/xgboost_prediction/"
MODEL_DIR = os.path.join(OUT_DIR, "models")
SCALER_PATH = os.path.join(OUT_DIR, "feature_scaler.joblib")
OUTPUT_XLSX = os.path.join(OUT_DIR, "predicted_formulation_properties.xlsx")

BO_EVALS = 60

for d in [OUT_DIR, MODEL_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


def load_train_data(path):
    """加载训练数据，返回 X_scaled, y_data, scaler, perf_names"""
    df = pd.read_excel(path)
    df_num = df.select_dtypes(include=['number']).dropna().replace([np.inf, -np.inf], np.nan).dropna()

    X_raw = df_num.iloc[:, :7].copy()
    perf_cols = df_num.columns[-4:]
    y_data = df_num[perf_cols].values

    scaler = StandardScaler().fit(X_raw.values)
    X_scaled = scaler.transform(X_raw.values)
    return X_scaled, y_data, scaler, perf_cols.tolist()


def main():

    print(">>> 正在加载训练数据...")
    X, y_data, scaler, perf_names = load_train_data(TRAIN_PATH)
    n_properties = y_data.shape[1]
    print(f">>> 性能指标: {perf_names}")
    print(f">>> 训练样本数: {X.shape[0]}")


    joblib.dump(scaler, SCALER_PATH)
    print(f">>> 缩放器已保存至: {SCALER_PATH}")


    space = {
        'max_depth': hp.quniform('max_depth', 3, 22, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
        'subsample': hp.uniform('subsample', 0.6, 0.9),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 0.9),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1)
    }

    best_params_all = {}


    for prop_idx in range(n_properties):
        y_raw = y_data[:, prop_idx]
        y_all = np.log1p(y_raw)
        prop_name = perf_names[prop_idx]

        print(f"\n{'='*60}")
        print(f">>> [{prop_idx+1}/{n_properties}] 性能指标: {prop_name}")
        print(f"{'='*60}")


        def objective(params):
            p = {
                'max_depth': int(params['max_depth']),
                'n_estimators': int(params['n_estimators']),
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'min_child_weight': int(params['min_child_weight']),
                'device': 'cuda',
                'random_state': RANDOM_STATE
            }
            model = xgb.XGBRegressor(**p)
            kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            score = np.mean(cross_val_score(model, X, y_all, cv=kf, scoring='r2'))
            return {'loss': -score, 'status': STATUS_OK}

        print(">>> 贝叶斯优化中...")
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=BO_EVALS,
            trials=trials,
            rstate=np.random.default_rng(RANDOM_STATE),
            show_progressbar=True
        )

        best_params = {
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate'],
            'n_estimators': int(best['n_estimators']),
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'min_child_weight': int(best['min_child_weight']),
            'device': 'cuda',
            'random_state': RANDOM_STATE
        }
        best_params_all[prop_name] = best_params
        print(f">>> 最优参数: {best_params}")


        print(f">>> 使用全部 {X.shape[0]} 条数据训练最终模型...")
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X, y_all)


        model_path = os.path.join(MODEL_DIR, f"xgb_model_{prop_name}.json")
        final_model.save_model(model_path)
        print(f">>> 模型已保存至: {model_path}")


        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_all, test_size=0.1, random_state=RANDOM_STATE
        )
        eval_model = xgb.XGBRegressor(**best_params)
        eval_model.fit(X_tr, y_tr)
        y_te_pred = eval_model.predict(X_te)
        r2_log = r2_score(y_te, y_te_pred)
        r2_orig = r2_score(np.expm1(y_te), np.expm1(y_te_pred))
        print(f"    - 测试集 R² (log空间): {r2_log:.4f}  |  原始空间: {r2_orig:.4f}")


    print(f"\n{'='*60}")
    print(">>> 加载公式空间进行大规模预测...")
    df_formula = pd.read_excel(PREDICT_PATH)
    X_formula_raw = df_formula.iloc[:, :7].values
    X_formula_scaled = scaler.transform(X_formula_raw)
    print(f">>> 公式空间样本数: {X_formula_scaled.shape[0]}")


    df_out = df_formula.iloc[:, :7].copy()

    for prop_name in perf_names:
        print(f">>> 预测中: {prop_name}...")
        model_path = os.path.join(MODEL_DIR, f"xgb_model_{prop_name}.json")
        model = xgb.XGBRegressor()
        model.load_model(model_path)

        y_pred_log = model.predict(X_formula_scaled)
        y_pred = np.expm1(y_pred_log)
        df_out[prop_name] = y_pred
        print(f"    - {prop_name} 预测完成, 范围: [{y_pred.min():.6f}, {y_pred.max():.6f}]")


    df_out.to_excel(OUTPUT_XLSX, index=False)
    print(f"\n>>> 配方性能库已保存至: {OUTPUT_XLSX}")
    print(f">>> 共计 {len(df_out)} 条配方, {len(perf_names)} 个性能指标")
    print(df_out.head(10).to_string())
    print(">>> 全部完成！")


if __name__ == "__main__":
    main()
