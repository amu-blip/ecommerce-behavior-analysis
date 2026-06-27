from pathlib import Path
import pandas as pd
import numpy as np
import joblib

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

# =====================================================
# 1. 定位项目路径
# =====================================================

current_file = Path(__file__).resolve()
project_dir = current_file.parents[1]

tables_dir = project_dir / "outputs" / "tables"
modeling_dir = project_dir / "data" / "modeling"
models_dir = project_dir / "models"

print("项目根目录：")
print(project_dir)

print("\n表格目录：")
print(tables_dir)

print("\n建模数据目录：")
print(modeling_dir)

print("\n模型目录：")
print(models_dir)

# =====================================================
# 2. 读取第 15 步 V2 输出结果
# =====================================================

metrics_path = tables_dir / "lightgbm_time_split_model_metrics.xlsx"
lift_path = tables_dir / "lightgbm_lift_analysis.xlsx"
importance_path = tables_dir / "lightgbm_feature_importance.xlsx"
confusion_path = tables_dir / "lightgbm_test_confusion_matrix.xlsx"
threshold_path = tables_dir / "lightgbm_threshold_search_valid.xlsx"

comparison_metrics_path = tables_dir / "model_v2_comparison_metrics.xlsx"
comparison_lift_path = tables_dir / "model_v2_comparison_lift.xlsx"

test_dataset_path = modeling_dir / "test_dataset.parquet"
feature_file = modeling_dir / "feature_columns.txt"
model_path = models_dir / "lightgbm_purchase_prediction_model.pkl"

required_files = [
    metrics_path,
    lift_path,
    importance_path,
    confusion_path,
    threshold_path,
    comparison_metrics_path,
    comparison_lift_path,
    test_dataset_path,
    feature_file,
    model_path
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(f"找不到必要文件：{file}")

metrics = pd.read_excel(metrics_path)
lift = pd.read_excel(lift_path)
importance = pd.read_excel(importance_path)
confusion = pd.read_excel(confusion_path)
threshold = pd.read_excel(threshold_path)

comparison_metrics = pd.read_excel(comparison_metrics_path)
comparison_lift = pd.read_excel(comparison_lift_path)

with open(feature_file, "r", encoding="utf-8") as f:
    feature_cols = [line.strip() for line in f.readlines() if line.strip()]

model = joblib.load(model_path)
test_df = pd.read_parquet(test_dataset_path)

target_col = "target_purchase"

print("\n最终 LightGBM 指标：")
print(metrics)

print("\n模型对比指标：")
print(comparison_metrics)

print("\n模型对比 Lift：")
print(comparison_lift)

print("\n特征重要性 Top 15：")
print(importance.head(15))

print("\n测试集数据规模：")
print(test_df.shape)

# =====================================================
# 3. 提取核心指标
# =====================================================

test_best = metrics[
    (metrics["dataset"] == "test_2020_04") &
    (metrics["threshold_name"] == "best_f1_threshold_from_valid")
].iloc[0]

valid_best = metrics[
    (metrics["dataset"] == "valid_2020_03") &
    (metrics["threshold_name"] == "best_f1_threshold_from_valid")
].iloc[0]

best_threshold = float(test_best["threshold_value"])

test_lift_10 = lift[
    (lift["dataset"] == "test_2020_04") &
    (lift["top_ratio"] == 0.10)
].iloc[0]

test_lift_5 = lift[
    (lift["dataset"] == "test_2020_04") &
    (lift["top_ratio"] == 0.05)
].iloc[0]

test_lift_1 = lift[
    (lift["dataset"] == "test_2020_04") &
    (lift["top_ratio"] == 0.01)
].iloc[0]

print("\n最终模型测试集最佳阈值指标：")
print(test_best)

print("\n最终模型测试集 Top 10% Lift：")
print(test_lift_10)

# =====================================================
# 4. 模型对比 / Ablation Study 解释表
# =====================================================

def get_best_metric_row(model_name):
    row = comparison_metrics[
        (comparison_metrics["model"] == model_name) &
        (comparison_metrics["dataset"] == "test_2020_04") &
        (comparison_metrics["threshold_name"] == "best_f1_threshold_from_valid")
    ]
    if len(row) == 0:
        return None
    return row.iloc[0]


def get_top10_lift_row(model_name):
    row = comparison_lift[
        (comparison_lift["model"] == model_name) &
        (comparison_lift["dataset"] == "test_2020_04") &
        (comparison_lift["top_ratio"] == 0.10)
    ]
    if len(row) == 0:
        return None
    return row.iloc[0]


model_names = [
    "baseline_has_cart_rule",
    "logistic_regression_full",
    "lgbm_behavior_only",
    "lgbm_behavior_price_time",
    "lgbm_full"
]

model_explain_name = {
    "baseline_has_cart_rule": "规则模型：是否加购",
    "logistic_regression_full": "逻辑回归：全部特征",
    "lgbm_behavior_only": "LightGBM：仅 session 行为特征",
    "lgbm_behavior_price_time": "LightGBM：行为 + 价格 + 时间特征",
    "lgbm_full": "LightGBM：全部特征"
}

ablation_rows = []

for model_name in model_names:
    metric_row = get_best_metric_row(model_name)
    lift_row = get_top10_lift_row(model_name)

    if metric_row is None or lift_row is None:
        continue

    ablation_rows.append({
        "model": model_name,
        "model_explanation": model_explain_name.get(model_name, model_name),
        "threshold": metric_row["threshold_value"],
        "accuracy": metric_row["accuracy"],
        "precision": metric_row["precision"],
        "recall": metric_row["recall"],
        "f1": metric_row["f1"],
        "roc_auc": metric_row["roc_auc"],
        "pr_auc": metric_row["pr_auc"],
        "top10_purchase_rate": lift_row["top_purchase_rate"],
        "top10_lift": lift_row["lift"],
        "top10_captured_purchase_ratio": lift_row["captured_purchase_ratio"]
    })

ablation_summary = pd.DataFrame(ablation_rows)

print("\nAblation Study 模型对比汇总：")
print(ablation_summary)

ablation_csv = tables_dir / "model_v2_ablation_summary.csv"
ablation_xlsx = tables_dir / "model_v2_ablation_summary.xlsx"

ablation_summary.to_csv(ablation_csv, index=False, encoding="utf-8-sig")
ablation_summary.to_excel(ablation_xlsx, index=False)

print("\nAblation Study 汇总表已保存：")
print(ablation_csv)
print(ablation_xlsx)

# =====================================================
# 5. 特征组重要性
# =====================================================

feature_group_map = {
    # session 加购行为
    "cart_count": "session_cart_behavior",
    "has_cart": "session_cart_behavior",
    "cart_view_ratio": "session_cart_behavior",

    # session 浏览与探索
    "view_count": "session_browse_behavior",
    "pre_purchase_event_count": "session_browse_behavior",
    "unique_product_count": "session_browse_behavior",
    "unique_category_count": "session_browse_behavior",
    "unique_main_category_count": "session_browse_behavior",
    "unique_brand_count": "session_browse_behavior",
    "events_per_product": "session_browse_behavior",

    # session 价格
    "avg_view_price": "session_price",
    "max_view_price": "session_price",
    "min_view_price": "session_price",
    "view_price_range": "session_price",
    "avg_cart_price": "session_price",
    "max_cart_price": "session_price",
    "min_cart_price": "session_price",
    "cart_price_range": "session_price",
    "cart_total_price": "session_price",
    "cart_to_view_price_ratio": "session_price",

    # session 时间
    "session_duration_seconds": "session_time",
    "start_hour": "session_time",
    "start_weekday_num": "session_time",

    # 用户历史活跃
    "user_hist_view_count": "user_history_activity",
    "user_hist_cart_count": "user_history_activity",
    "user_hist_active_day_count": "user_history_activity",
    "user_hist_view_product_count": "user_history_activity",
    "user_hist_cart_product_count": "user_history_activity",
    "user_hist_active_month_count": "user_history_activity",
    "user_hist_cart_view_ratio": "user_history_activity",

    # 用户历史购买
    "user_hist_purchase_count": "user_history_purchase",
    "user_hist_total_spend": "user_history_purchase",
    "user_hist_purchase_product_count": "user_history_purchase",
    "user_hist_purchase_month_count": "user_history_purchase",
    "user_hist_purchase_cart_ratio": "user_history_purchase",
    "user_hist_avg_spend_per_purchase": "user_history_purchase",
    "user_is_old_customer": "user_history_purchase"
}

feature_group_name = {
    "session_cart_behavior": "当前 session 加购行为",
    "session_browse_behavior": "当前 session 浏览/探索行为",
    "session_price": "当前 session 价格特征",
    "session_time": "当前 session 时间特征",
    "user_history_activity": "用户历史活跃特征",
    "user_history_purchase": "用户历史购买特征",
    "other": "其他"
}

importance_group = importance.copy()
importance_group["feature_group"] = importance_group["feature"].map(feature_group_map).fillna("other")
importance_group["feature_group_cn"] = importance_group["feature_group"].map(feature_group_name)

group_importance = importance_group.groupby(
    ["feature_group", "feature_group_cn"],
    as_index=False
).agg(
    importance_gain=("importance_gain", "sum"),
    feature_count=("feature", "count")
)

group_importance["importance_ratio"] = (
    group_importance["importance_gain"] /
    group_importance["importance_gain"].sum()
)

group_importance = group_importance.sort_values("importance_gain", ascending=False)

print("\n特征组重要性：")
print(group_importance)

group_importance_csv = tables_dir / "model_v2_feature_group_importance.csv"
group_importance_xlsx = tables_dir / "model_v2_feature_group_importance.xlsx"

group_importance.to_csv(group_importance_csv, index=False, encoding="utf-8-sig")
group_importance.to_excel(group_importance_xlsx, index=False)

print("\n特征组重要性表已保存：")
print(group_importance_csv)
print(group_importance_xlsx)

# =====================================================
# 6. 特征业务解释表
# =====================================================

feature_explanation_map = {
    "cart_count": "当前 session 中首次购买前的加购次数。加购次数越多，购买意向通常越强。",
    "has_cart": "当前 session 是否发生过加购。它是区分购买意向的重要行为信号。",
    "pre_purchase_event_count": "首次购买前的浏览和加购行为总数，反映购买前的行为深度。",
    "view_count": "当前 session 首次购买前的浏览次数，反映用户探索深度。",
    "unique_product_count": "当前 session 浏览或加购的不同商品数量，反映用户比较商品的范围。",
    "unique_category_count": "当前 session 涉及的不同品类数量，反映跨品类探索程度。",
    "unique_main_category_count": "当前 session 涉及的主品类数量，反映用户需求是否集中。",
    "unique_brand_count": "当前 session 涉及的品牌数量，反映品牌比较行为。",
    "avg_view_price": "浏览商品的平均价格，反映用户关注的价格水平。",
    "max_view_price": "浏览商品最高价格，反映用户接触到的价格上限。",
    "min_view_price": "浏览商品最低价格，反映用户接触到的价格下限。",
    "view_price_range": "浏览商品价格跨度，反映价格比较范围。",
    "avg_cart_price": "加购商品平均价格，反映购物车商品价值水平。",
    "max_cart_price": "加购商品最高价格，反映购物车中的高价商品信号。",
    "min_cart_price": "加购商品最低价格，反映购物车价格下限。",
    "cart_price_range": "加购商品价格跨度，反映购物车商品价格差异。",
    "cart_total_price": "当前 session 加购商品总金额，反映购物车价值。",
    "cart_view_ratio": "加购次数与浏览次数的比例，反映浏览向加购转化的强度。",
    "events_per_product": "平均每个商品对应的浏览/加购事件数，反映用户对单个商品的关注深度。",
    "cart_to_view_price_ratio": "加购商品均价与浏览商品均价的比例，反映用户最终加购商品价格相对浏览价格的变化。",
    "session_duration_seconds": "当前 session 持续时间，反映用户比较和决策时间。",
    "start_hour": "session 开始小时，反映一天内的访问时段差异。",
    "start_weekday_num": "session 开始星期，反映周内行为差异。",
    "user_hist_view_count": "用户历史浏览次数，反映长期活跃程度。",
    "user_hist_cart_count": "用户历史加购次数，反映过去的购买意向行为。",
    "user_hist_purchase_count": "用户历史购买次数，反映过去购买经验。",
    "user_hist_total_spend": "用户历史消费金额，反映用户价值。",
    "user_hist_active_day_count": "用户历史活跃天数，反映用户持续活跃程度。",
    "user_hist_view_product_count": "用户历史浏览过的不同商品数，反映长期兴趣范围。",
    "user_hist_cart_product_count": "用户历史加购过的不同商品数，反映过去加购兴趣范围。",
    "user_hist_purchase_product_count": "用户历史购买过的不同商品数，反映过去购买品类/商品丰富度。",
    "user_hist_active_month_count": "用户历史活跃月份数，反映用户长期活跃稳定性。",
    "user_hist_purchase_month_count": "用户历史发生购买的月份数，反映复购稳定性。",
    "user_hist_cart_view_ratio": "用户历史加购/浏览比例，反映历史行为转化倾向。",
    "user_hist_purchase_cart_ratio": "用户历史购买/加购比例，反映历史加购后成交倾向。",
    "user_hist_avg_spend_per_purchase": "用户历史单次购买平均消费金额，反映历史客单价。",
    "user_is_old_customer": "用户历史是否购买过，反映是否为老购买用户。"
}

feature_business_interpretation = importance_group.copy()
feature_business_interpretation["business_interpretation"] = feature_business_interpretation["feature"].map(
    feature_explanation_map
).fillna("该特征对模型预测有一定影响，可结合具体业务进一步解释。")

feature_business_interpretation = feature_business_interpretation[[
    "feature",
    "feature_group",
    "feature_group_cn",
    "importance_gain",
    "importance_ratio",
    "business_interpretation"
]]

feature_interpretation_csv = tables_dir / "feature_business_interpretation.csv"
feature_interpretation_xlsx = tables_dir / "feature_business_interpretation.xlsx"

feature_business_interpretation.to_csv(feature_interpretation_csv, index=False, encoding="utf-8-sig")
feature_business_interpretation.to_excel(feature_interpretation_xlsx, index=False)

print("\n特征业务解释表已保存：")
print(feature_interpretation_csv)
print(feature_interpretation_xlsx)

print("\n特征业务解释 Top 15：")
print(feature_business_interpretation.head(15))

# =====================================================
# 7. 测试集错误分析
# =====================================================

test_df[feature_cols] = test_df[feature_cols].fillna(0)
test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], 0)

y_test = test_df[target_col].astype(int)

try:
    test_proba = model.predict_proba(
        test_df[feature_cols],
        num_iteration=model.best_iteration_
    )[:, 1]
except TypeError:
    test_proba = model.predict_proba(test_df[feature_cols])[:, 1]

test_pred = (test_proba >= best_threshold).astype(int)

error_df = test_df.copy()
error_df["purchase_proba"] = test_proba
error_df["predicted_purchase"] = test_pred

def assign_error_type(row):
    if row[target_col] == 1 and row["predicted_purchase"] == 1:
        return "TP_true_purchase"
    elif row[target_col] == 0 and row["predicted_purchase"] == 1:
        return "FP_false_purchase"
    elif row[target_col] == 1 and row["predicted_purchase"] == 0:
        return "FN_missed_purchase"
    else:
        return "TN_true_no_purchase"

error_df["error_type"] = error_df.apply(assign_error_type, axis=1)

selected_error_features = [
    "purchase_proba",
    "view_count",
    "cart_count",
    "has_cart",
    "pre_purchase_event_count",
    "cart_view_ratio",
    "events_per_product",
    "avg_view_price",
    "avg_cart_price",
    "cart_total_price",
    "session_duration_seconds",
    "start_hour",
    "user_hist_view_count",
    "user_hist_cart_count",
    "user_hist_purchase_count",
    "user_hist_total_spend",
    "user_hist_active_month_count",
    "user_hist_purchase_cart_ratio",
    "user_is_old_customer"
]

selected_error_features = [col for col in selected_error_features if col in error_df.columns]

error_summary = error_df.groupby("error_type").agg(
    sample_count=("error_type", "count"),
    actual_purchase_rate=(target_col, "mean"),
    predicted_purchase_rate=("predicted_purchase", "mean")
).reset_index()

total_test_count = len(error_df)
error_summary["sample_ratio"] = error_summary["sample_count"] / total_test_count

feature_mean_by_error = error_df.groupby("error_type")[selected_error_features].mean().reset_index()

error_analysis = pd.merge(
    error_summary,
    feature_mean_by_error,
    on="error_type",
    how="left"
)

error_analysis = error_analysis.sort_values("sample_count", ascending=False)

print("\n测试集错误类型分析：")
print(error_analysis)

error_analysis_csv = tables_dir / "model_v2_error_analysis_by_group.csv"
error_analysis_xlsx = tables_dir / "model_v2_error_analysis_by_group.xlsx"

error_analysis.to_csv(error_analysis_csv, index=False, encoding="utf-8-sig")
error_analysis.to_excel(error_analysis_xlsx, index=False)

print("\n测试集错误分析表已保存：")
print(error_analysis_csv)
print(error_analysis_xlsx)

# =====================================================
# 8. 高分 session 画像
# =====================================================

profile_rows = []

for top_ratio in [0.01, 0.05, 0.10, 0.20]:
    top_n = int(len(error_df) * top_ratio)
    top_data = error_df.sort_values("purchase_proba", ascending=False).head(top_n)

    row = {
        "segment": f"top_{int(top_ratio * 100)}pct",
        "top_ratio": top_ratio,
        "sample_count": len(top_data),
        "actual_purchase_rate": top_data[target_col].mean(),
        "avg_purchase_proba": top_data["purchase_proba"].mean()
    }

    for col in selected_error_features:
        if col != "purchase_proba":
            row[f"avg_{col}"] = top_data[col].mean()

    profile_rows.append(row)

high_score_profile = pd.DataFrame(profile_rows)

print("\n高分 session 画像：")
print(high_score_profile)

profile_csv = tables_dir / "model_v2_high_score_session_profile.csv"
profile_xlsx = tables_dir / "model_v2_high_score_session_profile.xlsx"

high_score_profile.to_csv(profile_csv, index=False, encoding="utf-8-sig")
high_score_profile.to_excel(profile_xlsx, index=False)

print("\n高分 session 画像表已保存：")
print(profile_csv)
print(profile_xlsx)

# =====================================================
# 9. 最终模型解释汇总表
# =====================================================

baseline_metric = get_best_metric_row("baseline_has_cart_rule")
logistic_metric = get_best_metric_row("logistic_regression_full")
behavior_metric = get_best_metric_row("lgbm_behavior_only")
full_metric = get_best_metric_row("lgbm_full")

baseline_lift_10 = get_top10_lift_row("baseline_has_cart_rule")
logistic_lift_10 = get_top10_lift_row("logistic_regression_full")
behavior_lift_10 = get_top10_lift_row("lgbm_behavior_only")
full_lift_10 = get_top10_lift_row("lgbm_full")

model_summary_rows = [
    {
        "module": "模型任务",
        "item": "预测目标",
        "result": "预测一个 session 是否会产生购买行为",
        "interpretation": "将电商行为日志转化为 session 级二分类问题。"
    },
    {
        "module": "时间切分",
        "item": "训练 / 验证 / 测试",
        "result": "2019_10~2020_02 训练，2020_03 验证，2020_04 测试",
        "interpretation": "采用时间切分方式，更接近真实业务中用历史数据预测未来月份。"
    },
    {
        "module": "建模数据",
        "item": "特征工程",
        "result": "使用 37 个防泄漏 session 行为特征、价格特征、时间特征和用户历史特征",
        "interpretation": "购买 session 只使用首次 purchase 之前的 view/cart 构造特征，降低 session 内时间泄漏风险。"
    },
    {
        "module": "最终模型",
        "item": "模型选择",
        "result": "LightGBM full model",
        "interpretation": "在行为特征基础上加入用户历史特征后，整体 ROC-AUC、PR-AUC 和 Top 10% Lift 均表现最好。"
    },
    {
        "module": "最终模型",
        "item": "测试集 ROC-AUC / PR-AUC",
        "result": f"ROC-AUC={test_best['roc_auc']:.4f}, PR-AUC={test_best['pr_auc']:.4f}",
        "interpretation": "模型在类别不平衡场景下仍具备较强的排序和正类识别能力。"
    },
    {
        "module": "最终模型",
        "item": "测试集 Precision / Recall / F1",
        "result": f"Precision={test_best['precision']:.4f}, Recall={test_best['recall']:.4f}, F1={test_best['f1']:.4f}",
        "interpretation": "模型能召回多数真实购买 session，同时相比规则模型明显降低误报。"
    },
    {
        "module": "Lift 分析",
        "item": "Top 10% session",
        "result": f"购买率={test_lift_10['top_purchase_rate']:.2%}, Lift={test_lift_10['lift']:.2f}, 捕获真实购买={test_lift_10['captured_purchase_ratio']:.2%}",
        "interpretation": "模型更适合作为营销资源排序工具，而不是简单的买/不买硬分类器。"
    },
    {
        "module": "模型对比",
        "item": "相对 has_cart 规则 baseline",
        "result": f"Baseline Top10 Lift={baseline_lift_10['lift']:.2f}, Full Model Top10 Lift={full_lift_10['lift']:.2f}",
        "interpretation": "LightGBM full model 在简单加购规则基础上进一步提升了高价值 session 排序能力。"
    },
    {
        "module": "Ablation Study",
        "item": "用户历史特征贡献",
        "result": f"Behavior-only PR-AUC={behavior_metric['pr_auc']:.4f}, Full PR-AUC={full_metric['pr_auc']:.4f}",
        "interpretation": "加入用户历史特征后，模型对购买样本的识别能力进一步增强。"
    },
    {
        "module": "特征解释",
        "item": "最重要特征",
        "result": str(importance.iloc[0]["feature"]),
        "interpretation": "加购次数是购买预测中最核心的信号。"
    }
]

model_summary = pd.DataFrame(model_summary_rows)

summary_csv = tables_dir / "model_interpretation_summary.csv"
summary_xlsx = tables_dir / "model_interpretation_summary.xlsx"

model_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
model_summary.to_excel(summary_xlsx, index=False)

print("\n模型解释汇总表：")
print(model_summary)

print("\n模型解释汇总表已保存：")
print(summary_csv)
print(summary_xlsx)

# =====================================================
# 10. 生成业务建议文本
# =====================================================

business_text = f"""
LightGBM 购买预测模型 V2 解释与业务建议

一、模型任务与验证方式
本项目将用户行为日志聚合为 session 级建模数据，以 session 是否发生购买作为预测目标。模型采用时间切分方式进行验证：使用 2019_10 至 2020_02 作为训练集，2020_03 作为验证集，2020_04 作为测试集。相比随机划分，时间切分更接近真实业务中用历史数据预测未来月份的场景。

二、建模数据与防泄漏处理
新版建模数据集使用 37 个特征，覆盖当前 session 行为、价格、时间以及用户历史行为。为降低 session 内时间泄漏风险，对于发生购买的 session，特征只统计首次 purchase 之前的 view/cart 行为；对于未购买 session，则统计整个 session 中的 view/cart 行为。用户历史特征也只使用 session 所在月份之前的数据，避免使用未来行为。

三、最终模型表现
最终选择 LightGBM full model。该模型在 2020_04 测试集上取得 ROC-AUC={test_best['roc_auc']:.4f}，PR-AUC={test_best['pr_auc']:.4f}。在验证集选择的最佳 F1 阈值 {best_threshold:.2f} 下，测试集 Precision={test_best['precision']:.4f}，Recall={test_best['recall']:.4f}，F1={test_best['f1']:.4f}。

四、模型对比与 Ablation Study
has_cart 规则 baseline 在测试集上的 Top 10% Lift 为 {baseline_lift_10['lift']:.2f}，Logistic Regression 的 Top 10% Lift 为 {logistic_lift_10['lift']:.2f}，LightGBM full model 的 Top 10% Lift 达到 {full_lift_10['lift']:.2f}。这说明模型不仅学习到了加购行为这一强规则，还通过价格、时间和用户历史行为进一步提升了排序能力。

从 ablation study 看，仅使用 session 行为特征的 LightGBM 已经有较强表现；加入价格和时间特征后变化较小；加入用户历史特征后，full model 在 ROC-AUC、PR-AUC 和 Top 10% Lift 上表现最好。这说明用户历史活跃度和历史购买行为对购买预测具有增量价值。

五、Lift 分析
测试集整体购买率为 {test_lift_10['overall_purchase_rate']:.2%}。模型预测概率最高的前 10% session 中，真实购买率达到 {test_lift_10['top_purchase_rate']:.2%}，Lift 为 {test_lift_10['lift']:.2f} 倍，并覆盖了 {test_lift_10['captured_purchase_ratio']:.2%} 的真实购买 session。该结果说明模型非常适合用于营销资源排序，例如优惠券、客服触达、推荐位和购物车召回的优先级分配。

六、关键影响因素
特征重要性显示，cart_count 是最重要的预测变量，说明首次购买前的加购次数是判断购买意图的核心信号。has_cart、user_hist_active_month_count、user_hist_active_day_count、avg_cart_price、session_duration_seconds 等特征也具有较高重要性。这说明购买概率主要由当前 session 加购行为、用户历史活跃稳定性、购物车价格和 session 决策深度共同驱动。

七、错误分析
从错误类型看，False Positive 通常代表“模型认为有较强购买意向但实际未购买”的 session。这类用户往往已经有加购或较深浏览行为，可能由于价格、库存、物流、支付流程或临时比较等原因没有完成购买。False Negative 则代表“模型低估了购买意向但实际购买”的 session，可能包括快速决策、低浏览路径或老用户直接购买等场景。

八、业务建议
1. 将模型分数用于营销资源排序，而不是简单地把所有用户二分类为会买/不会买。
2. 对 Top 10% 高分 session 优先投放优惠券、限时折扣、库存提醒、客服触达或首页推荐位。
3. 对高分但未购买的 False Positive 类型用户重点做购物车召回，排查价格、库存、物流和支付流程障碍。
4. 对历史活跃月份数高、历史购买记录强的用户，可设置更高优先级的个性化推荐和会员运营策略。
5. 对加购次数高但未购买的用户，优先使用价格提醒、降价通知、优惠券或相似商品推荐。
6. 后续可将模型输出分数接入运营看板，用于每日高购买概率 session 排序和营销资源分配。
"""

business_text_path = tables_dir / "model_business_recommendations.txt"

with open(business_text_path, "w", encoding="utf-8") as f:
    f.write(business_text)

print("\n业务建议文本已保存：")
print(business_text_path)

print("\n业务建议文本预览：")
print(business_text)

print("\n第 16 步 V2：模型解释、错误分析与业务建议生成完成。")