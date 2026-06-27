from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

try:
    from lightgbm import LGBMClassifier
    from lightgbm import early_stopping, log_evaluation
except ImportError:
    raise ImportError(
        "未检测到 lightgbm。请先在 PyCharm 终端运行：pip install lightgbm"
    )

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# =====================================================
# 1. 定位项目路径
# =====================================================

current_file = Path(__file__).resolve()
project_dir = current_file.parents[1]

modeling_dir = project_dir / "data" / "modeling"
figures_dir = project_dir / "outputs" / "figures"
tables_dir = project_dir / "outputs" / "tables"
models_dir = project_dir / "models"

figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

print("项目根目录：")
print(project_dir)

print("\n建模数据目录：")
print(modeling_dir)

# =====================================================
# 2. 检查建模数据是否存在
# =====================================================

train_path = modeling_dir / "train_dataset.parquet"
valid_path = modeling_dir / "valid_dataset.parquet"
test_path = modeling_dir / "test_dataset.parquet"
feature_file = modeling_dir / "feature_columns.txt"

required_files = [
    train_path,
    valid_path,
    test_path,
    feature_file
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(f"找不到必要文件：{file}")

# =====================================================
# 3. 读取特征列
# =====================================================

with open(feature_file, "r", encoding="utf-8") as f:
    all_feature_cols = [line.strip() for line in f.readlines() if line.strip()]

target_col = "target_purchase"

print("\n特征数量：")
print(len(all_feature_cols))

print("\n特征列：")
for col in all_feature_cols:
    print(col)

# =====================================================
# 4. 读取 train / valid / test 数据
# =====================================================

print("\n开始读取训练集...")
train_df = pd.read_parquet(train_path)

print("开始读取验证集...")
valid_df = pd.read_parquet(valid_path)

print("开始读取测试集...")
test_df = pd.read_parquet(test_path)

print("\n训练集规模：")
print(train_df.shape)

print("\n验证集规模：")
print(valid_df.shape)

print("\n测试集规模：")
print(test_df.shape)

# =====================================================
# 5. 数据基础检查
# =====================================================

def check_dataset(df, name):
    print(f"\n{name} 目标变量分布：")
    print(df[target_col].value_counts())
    print(df[target_col].value_counts(normalize=True))

    missing_count = df[all_feature_cols].isna().sum().sum()
    print(f"\n{name} 特征缺失值总数：{missing_count}")

check_dataset(train_df, "训练集")
check_dataset(valid_df, "验证集")
check_dataset(test_df, "测试集")

# 缺失值统一填 0
train_df[all_feature_cols] = train_df[all_feature_cols].fillna(0)
valid_df[all_feature_cols] = valid_df[all_feature_cols].fillna(0)
test_df[all_feature_cols] = test_df[all_feature_cols].fillna(0)

# 防止极端情况下出现 inf
train_df[all_feature_cols] = train_df[all_feature_cols].replace([np.inf, -np.inf], 0)
valid_df[all_feature_cols] = valid_df[all_feature_cols].replace([np.inf, -np.inf], 0)
test_df[all_feature_cols] = test_df[all_feature_cols].replace([np.inf, -np.inf], 0)

y_train = train_df[target_col].astype(int)
y_valid = valid_df[target_col].astype(int)
y_test = test_df[target_col].astype(int)

# =====================================================
# 6. 定义特征组
# =====================================================
# 用于 ablation study：
# 1. 只使用 session 行为特征
# 2. session 行为 + 价格 + 时间
# 3. 使用全部 37 个特征

session_behavior_features = [
    "view_count",
    "cart_count",
    "has_cart",
    "pre_purchase_event_count",
    "unique_product_count",
    "unique_category_count",
    "unique_main_category_count",
    "unique_brand_count",
    "cart_view_ratio",
    "events_per_product"
]

price_features = [
    "avg_view_price",
    "max_view_price",
    "min_view_price",
    "view_price_range",
    "avg_cart_price",
    "max_cart_price",
    "min_cart_price",
    "cart_price_range",
    "cart_total_price",
    "cart_to_view_price_ratio"
]

time_features = [
    "session_duration_seconds",
    "start_hour",
    "start_weekday_num"
]

user_history_features = [
    "user_hist_view_count",
    "user_hist_cart_count",
    "user_hist_purchase_count",
    "user_hist_total_spend",
    "user_hist_active_day_count",
    "user_hist_view_product_count",
    "user_hist_cart_product_count",
    "user_hist_purchase_product_count",
    "user_hist_active_month_count",
    "user_hist_purchase_month_count",
    "user_hist_cart_view_ratio",
    "user_hist_purchase_cart_ratio",
    "user_hist_avg_spend_per_purchase",
    "user_is_old_customer"
]

# 只保留实际存在的列，避免以后特征文件变动时报错
def keep_existing_features(feature_list):
    return [col for col in feature_list if col in all_feature_cols]

session_behavior_features = keep_existing_features(session_behavior_features)
price_features = keep_existing_features(price_features)
time_features = keep_existing_features(time_features)
user_history_features = keep_existing_features(user_history_features)

feature_sets = {
    "lgbm_behavior_only": session_behavior_features,
    "lgbm_behavior_price_time": session_behavior_features + price_features + time_features,
    "lgbm_full": all_feature_cols
}

print("\n特征组：")
for name, cols in feature_sets.items():
    print(f"{name}: {len(cols)} features")

# =====================================================
# 7. 处理类别不平衡
# =====================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\n训练集负样本数：", negative_count)
print("训练集正样本数：", positive_count)
print("scale_pos_weight：", scale_pos_weight)

# =====================================================
# 8. 通用评估函数
# =====================================================

def get_basic_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "positive_prediction_ratio": float(np.mean(y_pred))
    }


def search_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []

    for threshold in thresholds:
        pred = (y_proba >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "positive_prediction_ratio": float(pred.mean())
        })

    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.sort_values("f1", ascending=False).iloc[0]

    return threshold_df, best_row


def lift_table(y_true, y_proba, dataset_name, model_name):
    temp = pd.DataFrame({
        "y_true": y_true.values,
        "y_proba": y_proba
    })

    temp = temp.sort_values("y_proba", ascending=False).reset_index(drop=True)

    overall_purchase_rate = temp["y_true"].mean()
    total_count = len(temp)
    total_purchase_count = temp["y_true"].sum()

    rows = []

    for top_ratio in [0.01, 0.05, 0.10, 0.20, 0.30]:
        top_n = int(total_count * top_ratio)
        top_data = temp.iloc[:top_n]

        top_purchase_rate = top_data["y_true"].mean()
        captured_purchase_count = top_data["y_true"].sum()

        rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "top_ratio": top_ratio,
            "top_n": top_n,
            "overall_purchase_rate": overall_purchase_rate,
            "top_purchase_rate": top_purchase_rate,
            "lift": top_purchase_rate / overall_purchase_rate if overall_purchase_rate > 0 else np.nan,
            "captured_purchase_count": captured_purchase_count,
            "total_purchase_count": total_purchase_count,
            "captured_purchase_ratio": captured_purchase_count / total_purchase_count if total_purchase_count > 0 else np.nan
        })

    return pd.DataFrame(rows)


def evaluate_model_output(model_name, feature_set_name, valid_proba, test_proba):
    valid_threshold_df, best_threshold_row = search_best_threshold(y_valid, valid_proba)
    best_threshold = float(best_threshold_row["threshold"])

    valid_pred_05 = (valid_proba >= 0.5).astype(int)
    test_pred_05 = (test_proba >= 0.5).astype(int)

    valid_pred_best = (valid_proba >= best_threshold).astype(int)
    test_pred_best = (test_proba >= best_threshold).astype(int)

    rows = []

    for dataset_name, y_true, y_pred, y_proba, threshold_name, threshold_value in [
        ("valid_2020_03", y_valid, valid_pred_05, valid_proba, "default_0.5", 0.5),
        ("test_2020_04", y_test, test_pred_05, test_proba, "default_0.5", 0.5),
        ("valid_2020_03", y_valid, valid_pred_best, valid_proba, "best_f1_threshold_from_valid", best_threshold),
        ("test_2020_04", y_test, test_pred_best, test_proba, "best_f1_threshold_from_valid", best_threshold)
    ]:
        metric = get_basic_metrics(y_true, y_pred, y_proba)
        metric.update({
            "model": model_name,
            "feature_set": feature_set_name,
            "dataset": dataset_name,
            "threshold_name": threshold_name,
            "threshold_value": threshold_value
        })
        rows.append(metric)

    metrics_df = pd.DataFrame(rows)

    valid_lift = lift_table(y_valid, valid_proba, "valid_2020_03", model_name)
    test_lift = lift_table(y_test, test_proba, "test_2020_04", model_name)
    lift_df = pd.concat([valid_lift, test_lift], ignore_index=True)

    return metrics_df, lift_df, valid_threshold_df, best_threshold


# =====================================================
# 9. Baseline 1：has_cart 规则模型
# =====================================================
# 规则：
# 如果 session 中出现加购，则预测会购买。
# 这个 baseline 很重要，因为它能证明 LightGBM 是否真的超过简单业务规则。

print("\n开始评估 Baseline：has_cart rule...")

valid_has_cart_score = valid_df["has_cart"].values.astype(float)
test_has_cart_score = test_df["has_cart"].values.astype(float)

baseline_metrics, baseline_lift, baseline_threshold_df, baseline_best_threshold = evaluate_model_output(
    model_name="baseline_has_cart_rule",
    feature_set_name="has_cart_only",
    valid_proba=valid_has_cart_score,
    test_proba=test_has_cart_score
)

print("\nBaseline has_cart 指标：")
print(baseline_metrics)

# =====================================================
# 10. Baseline 2：Logistic Regression
# =====================================================

RUN_LOGISTIC_BASELINE = True

all_metrics_list = [baseline_metrics]
all_lift_list = [baseline_lift]

logistic_model = None

if RUN_LOGISTIC_BASELINE:
    print("\n开始训练 Logistic Regression baseline...")

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    logistic_model.fit(train_df[all_feature_cols], y_train)

    valid_logistic_proba = logistic_model.predict_proba(valid_df[all_feature_cols])[:, 1]
    test_logistic_proba = logistic_model.predict_proba(test_df[all_feature_cols])[:, 1]

    logistic_metrics, logistic_lift, logistic_threshold_df, logistic_best_threshold = evaluate_model_output(
        model_name="logistic_regression_full",
        feature_set_name="all_features",
        valid_proba=valid_logistic_proba,
        test_proba=test_logistic_proba
    )

    print("\nLogistic Regression 指标：")
    print(logistic_metrics)

    all_metrics_list.append(logistic_metrics)
    all_lift_list.append(logistic_lift)

# =====================================================
# 11. LightGBM 训练函数
# =====================================================

def train_lgbm(feature_cols, model_name):
    print(f"\n开始训练 {model_name}...")
    print(f"使用特征数：{len(feature_cols)}")

    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",

        # early stopping 会自动选择最佳迭代轮数
        n_estimators=2000,
        learning_rate=0.03,

        num_leaves=63,
        max_depth=-1,
        min_child_samples=120,

        subsample=0.85,
        colsample_bytree=0.85,

        reg_alpha=0.1,
        reg_lambda=0.2,

        scale_pos_weight=scale_pos_weight,

        random_state=42,
        n_jobs=-1,
        importance_type="gain"
    )

    model.fit(
        train_df[feature_cols],
        y_train,
        eval_set=[(valid_df[feature_cols], y_valid)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=100)
        ]
    )

    print(f"{model_name} 训练完成。")
    print(f"best_iteration: {model.best_iteration_}")

    valid_proba = model.predict_proba(
        valid_df[feature_cols],
        num_iteration=model.best_iteration_
    )[:, 1]

    test_proba = model.predict_proba(
        test_df[feature_cols],
        num_iteration=model.best_iteration_
    )[:, 1]

    metrics_df, lift_df, threshold_df, best_threshold = evaluate_model_output(
        model_name=model_name,
        feature_set_name=model_name,
        valid_proba=valid_proba,
        test_proba=test_proba
    )

    print(f"\n{model_name} 指标：")
    print(metrics_df)

    return model, valid_proba, test_proba, metrics_df, lift_df, threshold_df, best_threshold

# =====================================================
# 12. Ablation Study：训练多个 LightGBM 特征组
# =====================================================

lgbm_models = {}
lgbm_outputs = {}

for model_name, feature_cols in feature_sets.items():
    model, valid_proba, test_proba, metrics_df, lift_df, threshold_df, best_threshold = train_lgbm(
        feature_cols=feature_cols,
        model_name=model_name
    )

    lgbm_models[model_name] = model

    lgbm_outputs[model_name] = {
        "feature_cols": feature_cols,
        "valid_proba": valid_proba,
        "test_proba": test_proba,
        "metrics_df": metrics_df,
        "lift_df": lift_df,
        "threshold_df": threshold_df,
        "best_threshold": best_threshold
    }

    all_metrics_list.append(metrics_df)
    all_lift_list.append(lift_df)

# =====================================================
# 13. 汇总模型对比结果
# =====================================================

model_comparison_metrics = pd.concat(all_metrics_list, ignore_index=True)
model_comparison_lift = pd.concat(all_lift_list, ignore_index=True)

# 为了阅读方便，调整列顺序
metric_cols_order = [
    "model",
    "feature_set",
    "dataset",
    "threshold_name",
    "threshold_value",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "positive_prediction_ratio"
]

model_comparison_metrics = model_comparison_metrics[metric_cols_order]

print("\n全部模型指标对比：")
print(model_comparison_metrics)

comparison_metrics_csv = tables_dir / "model_v2_comparison_metrics.csv"
comparison_metrics_xlsx = tables_dir / "model_v2_comparison_metrics.xlsx"

model_comparison_metrics.to_csv(comparison_metrics_csv, index=False, encoding="utf-8-sig")
model_comparison_metrics.to_excel(comparison_metrics_xlsx, index=False)

print("\n全部模型指标对比表已保存：")
print(comparison_metrics_csv)
print(comparison_metrics_xlsx)

comparison_lift_csv = tables_dir / "model_v2_comparison_lift.csv"
comparison_lift_xlsx = tables_dir / "model_v2_comparison_lift.xlsx"

model_comparison_lift.to_csv(comparison_lift_csv, index=False, encoding="utf-8-sig")
model_comparison_lift.to_excel(comparison_lift_xlsx, index=False)

print("\n全部模型 Lift 对比表已保存：")
print(comparison_lift_csv)
print(comparison_lift_xlsx)

# =====================================================
# 14. 选择最终模型
# =====================================================
# 默认选择 lgbm_full。
# 原因：
# 1. 使用全部 37 个特征；
# 2. 包含防泄漏 session 特征与用户历史特征；
# 3. 适合最终项目解释。
#
# 如果 ablation 结果显示其他模型测试集 Top 10% Lift 更好，也可以后续再调整。

final_model_name = "lgbm_full"

final_model = lgbm_models[final_model_name]
final_feature_cols = lgbm_outputs[final_model_name]["feature_cols"]
final_valid_proba = lgbm_outputs[final_model_name]["valid_proba"]
final_test_proba = lgbm_outputs[final_model_name]["test_proba"]
final_threshold_df = lgbm_outputs[final_model_name]["threshold_df"]
final_best_threshold = lgbm_outputs[final_model_name]["best_threshold"]

print("\n最终模型：")
print(final_model_name)

print("\n最终模型最佳阈值：")
print(final_best_threshold)

# =====================================================
# 15. 输出最终 LightGBM 指标
# =====================================================
# 为了兼容第 16 步，下面仍然保存旧文件名：
# lightgbm_time_split_model_metrics.xlsx
# lightgbm_lift_analysis.xlsx
# lightgbm_feature_importance.xlsx
# lightgbm_test_confusion_matrix.xlsx
# lightgbm_threshold_search_valid.xlsx

final_metrics = lgbm_outputs[final_model_name]["metrics_df"].copy()
final_lift = lgbm_outputs[final_model_name]["lift_df"].copy()

# 去掉 model / feature_set，保持第 16 步原来的读取逻辑兼容
canonical_metrics = final_metrics[[
    "dataset",
    "threshold_name",
    "threshold_value",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "positive_prediction_ratio"
]].copy()

canonical_lift = final_lift[[
    "dataset",
    "top_ratio",
    "top_n",
    "overall_purchase_rate",
    "top_purchase_rate",
    "lift",
    "captured_purchase_count",
    "total_purchase_count",
    "captured_purchase_ratio"
]].copy()

print("\n最终 LightGBM 指标：")
print(canonical_metrics)

metrics_csv = tables_dir / "lightgbm_time_split_model_metrics.csv"
metrics_xlsx = tables_dir / "lightgbm_time_split_model_metrics.xlsx"

canonical_metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
canonical_metrics.to_excel(metrics_xlsx, index=False)

print("\n最终 LightGBM 指标表已保存：")
print(metrics_csv)
print(metrics_xlsx)

threshold_csv = tables_dir / "lightgbm_threshold_search_valid.csv"
threshold_xlsx = tables_dir / "lightgbm_threshold_search_valid.xlsx"

final_threshold_df.to_csv(threshold_csv, index=False, encoding="utf-8-sig")
final_threshold_df.to_excel(threshold_xlsx, index=False)

print("\n最终 LightGBM 阈值搜索表已保存：")
print(threshold_csv)
print(threshold_xlsx)

lift_csv = tables_dir / "lightgbm_lift_analysis.csv"
lift_xlsx = tables_dir / "lightgbm_lift_analysis.xlsx"

canonical_lift.to_csv(lift_csv, index=False, encoding="utf-8-sig")
canonical_lift.to_excel(lift_xlsx, index=False)

print("\n最终 LightGBM Lift 表已保存：")
print(lift_csv)
print(lift_xlsx)

# =====================================================
# 16. 测试集分类报告与混淆矩阵
# =====================================================

test_pred_05 = (final_test_proba >= 0.5).astype(int)
test_pred_best = (final_test_proba >= final_best_threshold).astype(int)

print("\n最终 LightGBM 测试集分类报告，阈值 0.5：")
print(classification_report(y_test, test_pred_05, zero_division=0))

print("\n最终 LightGBM 测试集分类报告，最佳 F1 阈值：")
print(classification_report(y_test, test_pred_best, zero_division=0))

cm_05 = confusion_matrix(y_test, test_pred_05)
cm_best = confusion_matrix(y_test, test_pred_best)

cm_df = pd.DataFrame({
    "metric": [
        "TN_true_no_purchase",
        "FP_false_purchase",
        "FN_missed_purchase",
        "TP_true_purchase"
    ],
    "threshold_0_5": [
        cm_05[0, 0],
        cm_05[0, 1],
        cm_05[1, 0],
        cm_05[1, 1]
    ],
    "best_f1_threshold": [
        cm_best[0, 0],
        cm_best[0, 1],
        cm_best[1, 0],
        cm_best[1, 1]
    ]
})

print("\n最终 LightGBM 测试集混淆矩阵：")
print(cm_df)

cm_csv = tables_dir / "lightgbm_test_confusion_matrix.csv"
cm_xlsx = tables_dir / "lightgbm_test_confusion_matrix.xlsx"

cm_df.to_csv(cm_csv, index=False, encoding="utf-8-sig")
cm_df.to_excel(cm_xlsx, index=False)

print("\n最终 LightGBM 测试集混淆矩阵已保存：")
print(cm_csv)
print(cm_xlsx)

# =====================================================
# 17. 最终模型特征重要性
# =====================================================

feature_importance = pd.DataFrame({
    "feature": final_feature_cols,
    "importance_gain": final_model.feature_importances_
}).sort_values("importance_gain", ascending=False)

feature_importance["importance_ratio"] = (
    feature_importance["importance_gain"] /
    feature_importance["importance_gain"].sum()
)

print("\n最终 LightGBM 特征重要性：")
print(feature_importance)

importance_csv = tables_dir / "lightgbm_feature_importance.csv"
importance_xlsx = tables_dir / "lightgbm_feature_importance.xlsx"

feature_importance.to_csv(importance_csv, index=False, encoding="utf-8-sig")
feature_importance.to_excel(importance_xlsx, index=False)

print("\n最终 LightGBM 特征重要性表已保存：")
print(importance_csv)
print(importance_xlsx)

# =====================================================
# 18. 保存测试集预测结果样本
# =====================================================

test_result = test_df[[
    "dataset_type",
    "user_session",
    "user_id",
    "session_month",
    target_col
]].copy()

test_result["purchase_proba"] = final_test_proba
test_result["prediction_threshold_0_5"] = test_pred_05
test_result["prediction_best_threshold"] = test_pred_best

test_result = test_result.sort_values("purchase_proba", ascending=False)

test_result_sample = test_result.head(50000)

test_result_csv = tables_dir / "lightgbm_test_prediction_top50000.csv"
test_result_sample.to_csv(test_result_csv, index=False, encoding="utf-8-sig")

print("\n最终 LightGBM 测试集预测结果 Top 50000 已保存：")
print(test_result_csv)

# =====================================================
# 19. 保存最终模型
# =====================================================

model_path = models_dir / "lightgbm_purchase_prediction_model.pkl"
joblib.dump(final_model, model_path)

print("\n最终 LightGBM 模型文件已保存：")
print(model_path)

if logistic_model is not None:
    logistic_model_path = models_dir / "logistic_regression_purchase_prediction_model.pkl"
    joblib.dump(logistic_model, logistic_model_path)

    print("\nLogistic Regression baseline 模型文件已保存：")
    print(logistic_model_path)

# =====================================================
# 20. 绘图：最终模型 ROC 曲线
# =====================================================

fpr, tpr, _ = roc_curve(y_test, final_test_proba)
test_auc = roc_auc_score(y_test, final_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"LightGBM AUC={test_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

plt.title("ROC Curve - Test Month 2020_04")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()

roc_fig = figures_dir / "lightgbm_test_roc_curve.png"
plt.savefig(roc_fig, dpi=300)
plt.show()

print("\n最终模型 ROC 曲线图已保存：")
print(roc_fig)

# =====================================================
# 21. 绘图：最终模型 PR 曲线
# =====================================================

precision_values, recall_values, _ = precision_recall_curve(y_test, final_test_proba)
test_pr_auc = average_precision_score(y_test, final_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_values, precision_values, label=f"LightGBM PR-AUC={test_pr_auc:.4f}")

plt.title("Precision-Recall Curve - Test Month 2020_04")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()

pr_fig = figures_dir / "lightgbm_test_pr_curve.png"
plt.savefig(pr_fig, dpi=300)
plt.show()

print("\n最终模型 PR 曲线图已保存：")
print(pr_fig)

# =====================================================
# 22. 绘图：最终模型 Lift
# =====================================================

test_lift_plot = canonical_lift[canonical_lift["dataset"] == "test_2020_04"].copy()

plt.figure(figsize=(8, 5))
plt.bar(
    test_lift_plot["top_ratio"].astype(str),
    test_lift_plot["lift"]
)

plt.title("Lift by Top Predicted Probability - Test Month 2020_04")
plt.xlabel("Top Ratio")
plt.ylabel("Lift")

for i, value in enumerate(test_lift_plot["lift"]):
    plt.text(i, value, f"{value:.2f}", ha="center", va="bottom")

plt.tight_layout()

lift_fig = figures_dir / "lightgbm_test_lift.png"
plt.savefig(lift_fig, dpi=300)
plt.show()

print("\n最终模型 Lift 图已保存：")
print(lift_fig)

# =====================================================
# 23. 绘图：最终模型特征重要性 Top 20
# =====================================================

top_importance = feature_importance.head(20).copy()

plt.figure(figsize=(10, 7))
plt.barh(top_importance["feature"], top_importance["importance_gain"])

plt.title("LightGBM Feature Importance Top 20")
plt.xlabel("Importance Gain")
plt.ylabel("Feature")
plt.gca().invert_yaxis()

for i, value in enumerate(top_importance["importance_gain"]):
    plt.text(value, i, f"{value:.0f}", va="center")

plt.tight_layout()

importance_fig = figures_dir / "lightgbm_feature_importance_top20.png"
plt.savefig(importance_fig, dpi=300)
plt.show()

print("\n最终模型特征重要性 Top 20 图已保存：")
print(importance_fig)

# 兼容旧图名
importance_fig_old = figures_dir / "lightgbm_feature_importance_top15.png"
plt.figure(figsize=(10, 6))
top_importance_15 = feature_importance.head(15).copy()
plt.barh(top_importance_15["feature"], top_importance_15["importance_gain"])
plt.title("LightGBM Feature Importance Top 15")
plt.xlabel("Importance Gain")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(importance_fig_old, dpi=300)
plt.close()

# =====================================================
# 24. 绘图：模型对比 - 测试集 ROC-AUC / PR-AUC / Top10 Lift
# =====================================================

test_best_metrics = model_comparison_metrics[
    (model_comparison_metrics["dataset"] == "test_2020_04") &
    (model_comparison_metrics["threshold_name"] == "best_f1_threshold_from_valid")
].copy()

plt.figure(figsize=(10, 5))
plt.bar(test_best_metrics["model"], test_best_metrics["roc_auc"])

plt.title("Model Comparison - Test ROC-AUC")
plt.xlabel("Model")
plt.ylabel("ROC-AUC")
plt.xticks(rotation=30, ha="right")

for i, value in enumerate(test_best_metrics["roc_auc"]):
    plt.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

model_auc_fig = figures_dir / "model_v2_comparison_test_auc.png"
plt.savefig(model_auc_fig, dpi=300)
plt.show()

print("\n模型对比 ROC-AUC 图已保存：")
print(model_auc_fig)

plt.figure(figsize=(10, 5))
plt.bar(test_best_metrics["model"], test_best_metrics["pr_auc"])

plt.title("Model Comparison - Test PR-AUC")
plt.xlabel("Model")
plt.ylabel("PR-AUC")
plt.xticks(rotation=30, ha="right")

for i, value in enumerate(test_best_metrics["pr_auc"]):
    plt.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

model_pr_auc_fig = figures_dir / "model_v2_comparison_test_pr_auc.png"
plt.savefig(model_pr_auc_fig, dpi=300)
plt.show()

print("\n模型对比 PR-AUC 图已保存：")
print(model_pr_auc_fig)

test_top10_lift = model_comparison_lift[
    (model_comparison_lift["dataset"] == "test_2020_04") &
    (model_comparison_lift["top_ratio"] == 0.10)
].copy()

plt.figure(figsize=(10, 5))
plt.bar(test_top10_lift["model"], test_top10_lift["lift"])

plt.title("Model Comparison - Test Top 10% Lift")
plt.xlabel("Model")
plt.ylabel("Top 10% Lift")
plt.xticks(rotation=30, ha="right")

for i, value in enumerate(test_top10_lift["lift"]):
    plt.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

model_lift_fig = figures_dir / "model_v2_comparison_top10_lift.png"
plt.savefig(model_lift_fig, dpi=300)
plt.show()

print("\n模型对比 Top 10% Lift 图已保存：")
print(model_lift_fig)

# =====================================================
# 25. 输出关键结论辅助信息
# =====================================================

test_metric_best = canonical_metrics[
    (canonical_metrics["dataset"] == "test_2020_04") &
    (canonical_metrics["threshold_name"] == "best_f1_threshold_from_valid")
].iloc[0]

test_lift_10 = canonical_lift[
    (canonical_lift["dataset"] == "test_2020_04") &
    (canonical_lift["top_ratio"] == 0.10)
].iloc[0]

print("\n最终模型测试集最佳阈值指标：")
print(test_metric_best)

print("\n最终模型测试集 Top 10% Lift：")
print(test_lift_10)

print("\n最终模型 Top 15 特征重要性：")
print(feature_importance.head(15))

print("\n测试集不同模型最佳阈值表现对比：")
print(test_best_metrics.sort_values("roc_auc", ascending=False))

print("\n测试集不同模型 Top 10% Lift 对比：")
print(test_top10_lift.sort_values("lift", ascending=False))

print("\n第 15 步 V2：LightGBM 时间切分购买预测模型训练完成。")