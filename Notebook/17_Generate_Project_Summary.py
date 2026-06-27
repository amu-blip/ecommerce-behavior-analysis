from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

# =====================================================
# 1. 定位项目路径
# =====================================================

current_file = Path(__file__).resolve()
project_dir = current_file.parents[1]

outputs_dir = project_dir / "outputs"
tables_dir = outputs_dir / "tables"
figures_dir = outputs_dir / "figures"
modeling_dir = project_dir / "data" / "modeling"
models_dir = project_dir / "models"

readme_path = project_dir / "README.md"
summary_path = project_dir / "PROJECT_SUMMARY.md"
resume_path = project_dir / "RESUME_DESCRIPTION.md"
requirements_path = project_dir / "requirements.txt"
gitignore_path = project_dir / ".gitignore"

print("项目根目录：")
print(project_dir)

print("\n表格目录：")
print(tables_dir)

print("\n图表目录：")
print(figures_dir)

print("\n建模数据目录：")
print(modeling_dir)

print("\n模型目录：")
print(models_dir)

# =====================================================
# 2. 工具函数
# =====================================================

def read_table_if_exists(path: Path):
    if not path.exists():
        return None

    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    return None


def read_first_existing(paths):
    for path in paths:
        df = read_table_if_exists(path)
        if df is not None:
            print(f"读取成功：{path}")
            return df
    return None


def read_text_if_exists(path: Path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def fmt_int(value):
    if value is None or pd.isna(value):
        return "NA"
    return f"{int(round(float(value))):,}"


def fmt_float(value, digits=4):
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_number(value, digits=2):
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):,.{digits}f}"


def fmt_percent(value, digits=2):
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def safe_get(row, col, default=None):
    try:
        if row is None:
            return default
        if col not in row.index:
            return default
        value = row[col]
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def find_row(df, conditions):
    if df is None or len(df) == 0:
        return None

    temp = df.copy()

    for col, value in conditions.items():
        if col not in temp.columns:
            return None
        temp = temp[temp[col] == value]

    if len(temp) == 0:
        return None

    return temp.iloc[0]


def markdown_table(headers, rows):
    """
    不依赖 tabulate，手动生成 Markdown 表格。
    headers: list[str]
    rows: list[list[str]]
    """
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        safe_row = [str(item).replace("\n", " ") for item in row]
        lines.append("| " + " | ".join(safe_row) + " |")

    return "\n".join(lines)


code_fence = "```"
generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =====================================================
# 3. 读取项目结果表
# =====================================================

# 经营分析结果
monthly_kpi = read_first_existing([
    tables_dir / "monthly_kpi_summary.xlsx",
    tables_dir / "monthly_kpi_summary.csv",
    tables_dir / "multi_month_kpi_summary.xlsx",
    tables_dir / "multi_month_kpi_summary.csv",
])

# Cohort 结果
cohort_summary = read_first_existing([
    tables_dir / "cohort_summary.xlsx",
    tables_dir / "cohort_summary.csv",
])

monthly_new_repeat = read_first_existing([
    tables_dir / "monthly_new_repeat_purchase_users.xlsx",
    tables_dir / "monthly_new_repeat_purchase_users.csv",
])

# V2 建模数据结果
model_v2_dataset_check = read_first_existing([
    tables_dir / "model_v2_dataset_check.xlsx",
    tables_dir / "model_v2_dataset_check.csv",
])

model_v2_export_verify = read_first_existing([
    tables_dir / "model_v2_dataset_export_verify.xlsx",
    tables_dir / "model_v2_dataset_export_verify.csv",
])

# V2 模型结果
metrics = read_first_existing([
    tables_dir / "lightgbm_time_split_model_metrics.xlsx",
    tables_dir / "lightgbm_time_split_model_metrics.csv",
])

lift = read_first_existing([
    tables_dir / "lightgbm_lift_analysis.xlsx",
    tables_dir / "lightgbm_lift_analysis.csv",
])

feature_importance = read_first_existing([
    tables_dir / "lightgbm_feature_importance.xlsx",
    tables_dir / "lightgbm_feature_importance.csv",
])

ablation_summary = read_first_existing([
    tables_dir / "model_v2_ablation_summary.xlsx",
    tables_dir / "model_v2_ablation_summary.csv",
])

feature_group_importance = read_first_existing([
    tables_dir / "model_v2_feature_group_importance.xlsx",
    tables_dir / "model_v2_feature_group_importance.csv",
])

error_analysis = read_first_existing([
    tables_dir / "model_v2_error_analysis_by_group.xlsx",
    tables_dir / "model_v2_error_analysis_by_group.csv",
])

high_score_profile = read_first_existing([
    tables_dir / "model_v2_high_score_session_profile.xlsx",
    tables_dir / "model_v2_high_score_session_profile.csv",
])

model_interpretation = read_first_existing([
    tables_dir / "model_interpretation_summary.xlsx",
    tables_dir / "model_interpretation_summary.csv",
])

business_recommendations_text = read_text_if_exists(
    tables_dir / "model_business_recommendations.txt"
)

# =====================================================
# 4. 提取核心指标，缺失时使用项目已知结果兜底
# =====================================================

# 数据规模：使用 V2 全量检查结果
total_event_count = 410_995_046
total_month_count = 7
total_day_count = 213
total_user_count = 15_635_832
total_session_count = 89_614_715

# 购买规模：来自 Cohort / 购买分析
total_purchase_event_count = 6_848_824
total_purchase_user_count = 2_064_899

# 第 14 步 V2 建模数据规模
train_row_count = 1_310_243
valid_row_count = 378_088
test_row_count = 348_290

train_purchase_ratio = 0.057624
valid_purchase_ratio = 0.066670
test_purchase_ratio = 0.067777

if model_v2_export_verify is not None:
    train_row = find_row(model_v2_export_verify, {"dataset_type": "train"})
    valid_row = find_row(model_v2_export_verify, {"dataset_type": "valid"})
    test_row = find_row(model_v2_export_verify, {"dataset_type": "test"})

    train_row_count = safe_get(train_row, "row_count", train_row_count)
    valid_row_count = safe_get(valid_row, "row_count", valid_row_count)
    test_row_count = safe_get(test_row, "row_count", test_row_count)

    train_purchase_ratio = safe_get(train_row, "purchase_ratio", train_purchase_ratio)
    valid_purchase_ratio = safe_get(valid_row, "purchase_ratio", valid_purchase_ratio)
    test_purchase_ratio = safe_get(test_row, "purchase_ratio", test_purchase_ratio)

# 月度 KPI
best_revenue_month = "2020_02"
best_revenue = 381_178_900
best_purchase_count = 1_200_288
best_purchase_view_ratio = 0.023468

if monthly_kpi is not None and len(monthly_kpi) > 0:
    month_col = "event_month" if "event_month" in monthly_kpi.columns else None

    if month_col and "revenue" in monthly_kpi.columns:
        best_revenue_row = monthly_kpi.sort_values("revenue", ascending=False).iloc[0]
        best_revenue_month = best_revenue_row[month_col]
        best_revenue = best_revenue_row["revenue"]

    if "purchase_count" in monthly_kpi.columns:
        best_purchase_row = monthly_kpi.sort_values("purchase_count", ascending=False).iloc[0]
        best_purchase_count = best_purchase_row["purchase_count"]

    if "purchase_view_ratio" in monthly_kpi.columns:
        best_ratio_row = monthly_kpi.sort_values("purchase_view_ratio", ascending=False).iloc[0]
        best_purchase_view_ratio = best_ratio_row["purchase_view_ratio"]

# Cohort
best_cohort_month = "2019_10"
best_month_1_retention = 0.262983
largest_cohort_month = "2019_11"
largest_cohort_user_count = 350_352

if cohort_summary is not None and len(cohort_summary) > 0:
    if "month_1_retention_rate" in cohort_summary.columns:
        valid_cohort = cohort_summary[cohort_summary["month_1_retention_rate"] > 0].copy()
        if len(valid_cohort) > 0:
            best_retention_row = valid_cohort.sort_values("month_1_retention_rate", ascending=False).iloc[0]
            best_cohort_month = safe_get(best_retention_row, "cohort_month", best_cohort_month)
            best_month_1_retention = safe_get(best_retention_row, "month_1_retention_rate", best_month_1_retention)

    if "cohort_user_count" in cohort_summary.columns:
        largest_cohort_row = cohort_summary.sort_values("cohort_user_count", ascending=False).iloc[0]
        largest_cohort_month = safe_get(largest_cohort_row, "cohort_month", largest_cohort_month)
        largest_cohort_user_count = safe_get(largest_cohort_row, "cohort_user_count", largest_cohort_user_count)

best_repeat_month = "2020_03"
best_repeat_ratio = 0.429589

if monthly_new_repeat is not None and len(monthly_new_repeat) > 0:
    if "repeat_user_ratio" in monthly_new_repeat.columns:
        best_repeat_row = monthly_new_repeat.sort_values("repeat_user_ratio", ascending=False).iloc[0]
        best_repeat_month = safe_get(best_repeat_row, "event_month", best_repeat_month)
        best_repeat_ratio = safe_get(best_repeat_row, "repeat_user_ratio", best_repeat_ratio)

# 最终模型指标
test_auc = 0.971755
test_pr_auc = 0.634176
test_accuracy = 0.942792
test_precision = 0.557306
test_recall = 0.758239
test_f1 = 0.642428
best_threshold = 0.49

test_best = find_row(metrics, {
    "dataset": "test_2020_04",
    "threshold_name": "best_f1_threshold_from_valid"
})

if test_best is not None:
    test_auc = safe_get(test_best, "roc_auc", test_auc)
    test_pr_auc = safe_get(test_best, "pr_auc", test_pr_auc)
    test_accuracy = safe_get(test_best, "accuracy", test_accuracy)
    test_precision = safe_get(test_best, "precision", test_precision)
    test_recall = safe_get(test_best, "recall", test_recall)
    test_f1 = safe_get(test_best, "f1", test_f1)
    best_threshold = safe_get(test_best, "threshold_value", best_threshold)

# Lift
overall_purchase_rate = 0.067777
top1_purchase_rate = 0.774555
top5_purchase_rate = 0.662857
top10_purchase_rate = 0.540326
top10_lift = 7.972126
top10_captured_purchase_ratio = 0.797213
top10_captured_purchase_count = 18_819
test_total_purchase_count = 23_606

if lift is not None:
    top1_row = find_row(lift, {"dataset": "test_2020_04", "top_ratio": 0.01})
    top5_row = find_row(lift, {"dataset": "test_2020_04", "top_ratio": 0.05})
    top10_row = find_row(lift, {"dataset": "test_2020_04", "top_ratio": 0.10})

    if top1_row is not None:
        top1_purchase_rate = safe_get(top1_row, "top_purchase_rate", top1_purchase_rate)

    if top5_row is not None:
        top5_purchase_rate = safe_get(top5_row, "top_purchase_rate", top5_purchase_rate)

    if top10_row is not None:
        overall_purchase_rate = safe_get(top10_row, "overall_purchase_rate", overall_purchase_rate)
        top10_purchase_rate = safe_get(top10_row, "top_purchase_rate", top10_purchase_rate)
        top10_lift = safe_get(top10_row, "lift", top10_lift)
        top10_captured_purchase_ratio = safe_get(top10_row, "captured_purchase_ratio", top10_captured_purchase_ratio)
        top10_captured_purchase_count = safe_get(top10_row, "captured_purchase_count", top10_captured_purchase_count)
        test_total_purchase_count = safe_get(top10_row, "total_purchase_count", test_total_purchase_count)

# Ablation
baseline_top10_lift = 6.355164
logistic_top10_lift = 7.074473
behavior_only_top10_lift = 7.917055
behavior_price_time_top10_lift = 7.889943
full_model_top10_lift = top10_lift

behavior_only_pr_auc = 0.613305
full_model_pr_auc = test_pr_auc

if ablation_summary is not None:
    baseline_row = find_row(ablation_summary, {"model": "baseline_has_cart_rule"})
    logistic_row = find_row(ablation_summary, {"model": "logistic_regression_full"})
    behavior_row = find_row(ablation_summary, {"model": "lgbm_behavior_only"})
    behavior_price_row = find_row(ablation_summary, {"model": "lgbm_behavior_price_time"})
    full_row = find_row(ablation_summary, {"model": "lgbm_full"})

    baseline_top10_lift = safe_get(baseline_row, "top10_lift", baseline_top10_lift)
    logistic_top10_lift = safe_get(logistic_row, "top10_lift", logistic_top10_lift)
    behavior_only_top10_lift = safe_get(behavior_row, "top10_lift", behavior_only_top10_lift)
    behavior_price_time_top10_lift = safe_get(behavior_price_row, "top10_lift", behavior_price_time_top10_lift)
    full_model_top10_lift = safe_get(full_row, "top10_lift", full_model_top10_lift)

    behavior_only_pr_auc = safe_get(behavior_row, "pr_auc", behavior_only_pr_auc)
    full_model_pr_auc = safe_get(full_row, "pr_auc", full_model_pr_auc)

# 特征重要性
top_feature = "cart_count"
top_feature_ratio = 0.789645

if feature_importance is not None and len(feature_importance) > 0:
    top_feature = feature_importance.iloc[0]["feature"]
    top_feature_ratio = feature_importance.iloc[0]["importance_ratio"]

# 特征组重要性
top_feature_group_cn = "当前 session 加购行为"
top_feature_group_ratio = 0.861636

if feature_group_importance is not None and len(feature_group_importance) > 0:
    top_group_row = feature_group_importance.sort_values("importance_ratio", ascending=False).iloc[0]
    top_feature_group_cn = safe_get(top_group_row, "feature_group_cn", top_feature_group_cn)
    top_feature_group_ratio = safe_get(top_group_row, "importance_ratio", top_feature_group_ratio)

# 错误分析
fp_count = 14_218
fn_count = 5_707
tp_count = 17_899
tn_count = 310_466

if error_analysis is not None:
    fp_row = find_row(error_analysis, {"error_type": "FP_false_purchase"})
    fn_row = find_row(error_analysis, {"error_type": "FN_missed_purchase"})
    tp_row = find_row(error_analysis, {"error_type": "TP_true_purchase"})
    tn_row = find_row(error_analysis, {"error_type": "TN_true_no_purchase"})

    fp_count = safe_get(fp_row, "sample_count", fp_count)
    fn_count = safe_get(fn_row, "sample_count", fn_count)
    tp_count = safe_get(tp_row, "sample_count", tp_count)
    tn_count = safe_get(tn_row, "sample_count", tn_count)

# =====================================================
# 5. 生成 Markdown 片段
# =====================================================

data_scale_table = markdown_table(
    headers=["指标", "数值"],
    rows=[
        ["时间范围", "2019-10 至 2020-04"],
        ["月份数", fmt_int(total_month_count)],
        ["天数", fmt_int(total_day_count)],
        ["有效行为记录数", fmt_int(total_event_count)],
        ["用户数", fmt_int(total_user_count)],
        ["Session 数", fmt_int(total_session_count)],
        ["购买事件数", fmt_int(total_purchase_event_count)],
        ["购买用户数", fmt_int(total_purchase_user_count)],
    ]
)

model_dataset_table = markdown_table(
    headers=["数据集", "时间范围", "样本数", "购买样本占比"],
    rows=[
        ["Train", "2019-10 至 2020-02", fmt_int(train_row_count), fmt_percent(train_purchase_ratio)],
        ["Valid", "2020-03", fmt_int(valid_row_count), fmt_percent(valid_purchase_ratio)],
        ["Test", "2020-04", fmt_int(test_row_count), fmt_percent(test_purchase_ratio)],
    ]
)

model_metric_table = markdown_table(
    headers=["指标", "测试集结果"],
    rows=[
        ["ROC-AUC", fmt_float(test_auc, 4)],
        ["PR-AUC", fmt_float(test_pr_auc, 4)],
        ["Best F1 Threshold", fmt_float(best_threshold, 2)],
        ["Accuracy", fmt_float(test_accuracy, 4)],
        ["Precision", fmt_float(test_precision, 4)],
        ["Recall", fmt_float(test_recall, 4)],
        ["F1-score", fmt_float(test_f1, 4)],
    ]
)

lift_table_md = markdown_table(
    headers=["分组", "购买率", "Lift", "捕获真实购买比例"],
    rows=[
        ["整体测试集", fmt_percent(overall_purchase_rate), "1.00", "100.00%"],
        ["Top 1% 高分 session", fmt_percent(top1_purchase_rate), "NA", "NA"],
        ["Top 5% 高分 session", fmt_percent(top5_purchase_rate), "NA", "NA"],
        ["Top 10% 高分 session", fmt_percent(top10_purchase_rate), fmt_float(top10_lift, 2), fmt_percent(top10_captured_purchase_ratio)],
    ]
)

ablation_table = markdown_table(
    headers=["模型", "说明", "Top 10% Lift"],
    rows=[
        ["has_cart rule", "只根据是否加购判断购买倾向", fmt_float(baseline_top10_lift, 2)],
        ["Logistic Regression", "线性模型，使用全部特征", fmt_float(logistic_top10_lift, 2)],
        ["LightGBM behavior only", "只使用当前 session 行为特征", fmt_float(behavior_only_top10_lift, 2)],
        ["LightGBM behavior + price + time", "加入价格和时间特征", fmt_float(behavior_price_time_top10_lift, 2)],
        ["LightGBM full", "加入用户历史特征，最终模型", fmt_float(full_model_top10_lift, 2)],
    ]
)

error_table = markdown_table(
    headers=["错误类型", "样本数", "含义"],
    rows=[
        ["TN", fmt_int(tn_count), "实际未购买，模型也判断为未购买"],
        ["TP", fmt_int(tp_count), "实际购买，模型也判断为购买"],
        ["FP", fmt_int(fp_count), "模型判断会购买，但实际未购买"],
        ["FN", fmt_int(fn_count), "模型判断不会购买，但实际购买"],
    ]
)

# 特征组表
feature_group_rows = []

if feature_group_importance is not None and len(feature_group_importance) > 0:
    temp_group = feature_group_importance.sort_values("importance_ratio", ascending=False)

    for _, row in temp_group.iterrows():
        feature_group_rows.append([
            safe_get(row, "feature_group_cn", "NA"),
            fmt_int(safe_get(row, "feature_count", 0)),
            fmt_percent(safe_get(row, "importance_ratio", 0)),
        ])
else:
    feature_group_rows = [
        ["当前 session 加购行为", "3", "86.16%"],
        ["用户历史活跃特征", "7", "5.68%"],
        ["当前 session 价格特征", "10", "2.66%"],
        ["用户历史购买特征", "7", "2.11%"],
        ["当前 session 浏览/探索行为", "7", "1.80%"],
        ["当前 session 时间特征", "3", "1.59%"],
    ]

feature_group_table = markdown_table(
    headers=["特征组", "特征数", "重要性占比"],
    rows=feature_group_rows
)

# Top 特征表
top_feature_rows = []

if feature_importance is not None and len(feature_importance) > 0:
    temp_importance = feature_importance.head(10)

    for _, row in temp_importance.iterrows():
        top_feature_rows.append([
            safe_get(row, "feature", "NA"),
            fmt_percent(safe_get(row, "importance_ratio", 0)),
        ])
else:
    top_feature_rows = [
        ["cart_count", "78.96%"],
        ["has_cart", "7.00%"],
        ["user_hist_active_month_count", "2.90%"],
        ["user_hist_active_day_count", "2.41%"],
        ["avg_cart_price", "1.66%"],
    ]

top_feature_table = markdown_table(
    headers=["特征", "重要性占比"],
    rows=top_feature_rows
)

# =====================================================
# 6. 生成 README.md
# =====================================================

readme_text = f"""# 多月电商用户行为分析与购买转化预测

## 1. 项目概述

本项目基于多品类电商平台用户行为日志，围绕 `view`、`cart`、`purchase` 三类核心行为，完成从大规模数据处理、经营指标分析、用户留存分析到 session 级购买预测建模的完整流程。

项目重点不是单纯做 EDA，而是构建一条可复用的数据分析与建模链路：

{code_fence}text
Raw CSV files
    ↓
DuckDB 数据清洗
    ↓
按月份 / 日期导出 Parquet 分区数据
    ↓
多月经营 KPI 分析
    ↓
Cohort 留存与复购分析
    ↓
防泄漏 session 级特征工程
    ↓
LightGBM 购买预测模型
    ↓
模型解释、错误分析与业务建议
{code_fence}

## 2. 数据规模

{data_scale_table}

本项目使用 DuckDB 处理大规模行为日志，并将清洗后的数据导出为按月份和日期分区的 Parquet 文件，降低后续分析和建模的重复读取成本。

## 3. 技术栈

- Python
- DuckDB
- pandas
- NumPy
- scikit-learn
- LightGBM
- matplotlib
- Parquet
- openpyxl

## 4. 项目结构

{code_fence}text
Ecommerce_behavior_project/
├── Data/
│   └── Processed/
│       ├── 2019_10/
│       ├── 2019_11/
│       ├── 2019_12/
│       ├── 2020_01/
│       ├── 2020_02/
│       ├── 2020_03/
│       └── 2020_04/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── modeling/
│       ├── train_dataset.parquet
│       ├── valid_dataset.parquet
│       ├── test_dataset.parquet
│       └── feature_columns.txt
│
├── Notebook/
│   ├── 11_Export_All_Months_Parquet.py
│   ├── 12_Multi_Month_KPI_Trend.py
│   ├── 13_Cohort_Retention_Analysis.py
│   ├── 14_Build_Model_Dataset_DuckDB.py
│   ├── 15_Train_LightGBM_Time_Split.py
│   ├── 16_Model_Evaluation_And_Interpretation.py
│   └── 17_Generate_Project_Summary.py
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── models/
│   └── lightgbm_purchase_prediction_model.pkl
│
├── README.md
├── PROJECT_SUMMARY.md
├── RESUME_DESCRIPTION.md
├── requirements.txt
└── .gitignore
{code_fence}

## 5. 数据工程

原始数据为多个大体量 CSV 文件，不适合在每个分析脚本中反复使用 pandas 直接读取。因此，本项目使用 DuckDB 对原始行为日志进行清洗，并导出为 Parquet 分区数据。

Parquet 目录按月份和日期组织：

{code_fence}text
Data/Processed/
├── 2019_10/
│   ├── 10.1/
│   ├── 10.2/
│   └── ...
├── 2019_11/
├── 2019_12/
├── 2020_01/
├── 2020_02/
├── 2020_03/
└── 2020_04/
{code_fence}

该结构的优势是：

1. 避免反复扫描原始 CSV。
2. 支持按月份、日期灵活读取。
3. 提高大规模聚合分析效率。
4. 为后续建模数据集构建提供稳定数据层。

## 6. 多月经营 KPI 分析

项目构建了月度经营指标体系，包括：

- 总行为数
- 浏览数
- 加购数
- 购买数
- 销售额
- 活跃用户数
- 购买用户数
- session 数
- 购买 session 数
- 购买 / 浏览比例
- 平均购买价格
- 人均购买金额

核心发现：

- 销售额最高月份为 **{best_revenue_month}**，销售额约 **{fmt_number(best_revenue, 2)}**。
- 购买次数最高月份购买事件数约 **{fmt_int(best_purchase_count)}**。
- 最高购买 / 浏览比例约为 **{fmt_percent(best_purchase_view_ratio)}**。

这些指标用于识别不同月份的交易峰值、转化效率变化和用户购买行为变化。

## 7. Cohort 留存与复购分析

项目基于用户首次购买月份构建 cohort 留存矩阵，分析不同首购月份用户在后续月份的复购表现。

核心发现：

- 次月留存率最高的 cohort 为 **{best_cohort_month}**，次月留存率达到 **{fmt_percent(best_month_1_retention)}**。
- 首购用户规模最大的 cohort 为 **{largest_cohort_month}**，首购用户数为 **{fmt_int(largest_cohort_user_count)}**。
- 复购用户占比最高月份为 **{best_repeat_month}**，复购用户占比达到 **{fmt_percent(best_repeat_ratio)}**。

该部分说明，用户规模和用户质量并不完全一致，首购用户数量高并不必然意味着后续留存更好。

## 8. Session 级购买预测建模

### 8.1 建模目标

将用户行为日志聚合为 session 级样本，预测一个 session 是否会发生购买：

{code_fence}text
target_purchase = 该 session 是否出现 purchase 行为
{code_fence}

这是一个典型的类别不平衡二分类问题。测试集中购买 session 占比约为 **{fmt_percent(test_purchase_ratio)}**。

### 8.2 时间切分

为模拟真实业务中“用历史数据预测未来月份”的场景，项目采用时间切分，而不是随机切分。

{model_dataset_table}

### 8.3 防泄漏特征工程

新版建模数据集使用 **37 个特征**，覆盖四类信息：

1. 当前 session 行为特征。
2. 当前 session 价格特征。
3. 当前 session 时间特征。
4. 用户历史行为特征。

为降低时间泄漏风险，项目进行了两层约束：

- 对于发生购买的 session，只使用首次 `purchase` 之前的 `view/cart` 行为构造特征。
- 用户历史特征只使用 session 所在月份之前的数据，不使用当前月未来行为。

示例：

{code_fence}text
view -> cart -> purchase -> view -> cart
{code_fence}

在构造特征时，只使用：

{code_fence}text
view -> cart
{code_fence}

购买后的行为不会进入模型特征。

## 9. 模型训练与对比

项目训练并比较了多个模型：

{ablation_table}

最终选择 **LightGBM full model**，因为它在测试集上的 ROC-AUC、PR-AUC、F1 和 Top 10% Lift 综合表现最好。

## 10. 最终模型表现

最终模型在 2020-04 测试集上的表现如下：

{model_metric_table}

由于购买样本占比较低，项目不仅关注 accuracy，也重点关注 PR-AUC、Recall、F1 和 Lift。

## 11. Lift 分析

在电商营销场景中，模型的排序能力通常比单纯二分类更重要。本项目使用 Lift 分析衡量模型识别高购买概率 session 的能力。

{lift_table_md}

核心结论：

- 测试集整体购买率为 **{fmt_percent(overall_purchase_rate)}**。
- 模型预测概率最高的前 10% session 中，真实购买率达到 **{fmt_percent(top10_purchase_rate)}**。
- Top 10% Lift 为 **{fmt_float(top10_lift, 2)} 倍**。
- Top 10% 高分 session 覆盖了 **{fmt_percent(top10_captured_purchase_ratio)}** 的真实购买 session。

这说明模型适合用于营销资源排序，例如优惠券、客服触达、推荐位分配和购物车召回优先级排序。

## 12. 特征重要性

### 12.1 特征组重要性

{feature_group_table}

当前 session 加购行为是最重要的特征组，重要性占比约 **{fmt_percent(top_feature_group_ratio)}**。这说明购买预测主要由加购行为驱动，同时用户历史活跃度、价格特征和历史购买行为也提供了增量信息。

### 12.2 Top 特征

{top_feature_table}

最重要的单个特征是 **{top_feature}**，说明首次购买前的加购次数是购买意向最核心的行为信号。

## 13. 错误分析

使用验证集选择的最佳 F1 阈值 **{fmt_float(best_threshold, 2)}** 后，测试集错误类型如下：

{error_table}

错误分析解释：

- False Positive：模型判断会购买，但实际未购买。这类 session 往往已经出现较强加购或浏览信号，可能受价格、库存、物流、支付流程或临时比较影响而没有成交。
- False Negative：模型判断不会购买，但实际购买。这类 session 可能包括快速决策、低浏览路径、老用户直接购买等场景。

错误分析说明，模型更适合作为营销资源排序工具，而不是绝对的“买 / 不买”判断器。

## 14. 业务建议

基于 KPI、Cohort、模型预测和错误分析，提出以下业务建议：

1. 将模型分数用于营销资源排序，而不是简单地把用户二分类为会买或不会买。
2. 对 Top 10% 高购买概率 session 优先投放优惠券、限时折扣、库存提醒、客服触达或推荐位。
3. 对加购但未购买用户重点做购物车召回，排查价格、库存、物流和支付流程障碍。
4. 对历史活跃月份数高、历史购买记录强的用户，设置更高优先级的个性化推荐和会员运营策略。
5. 对高购物车金额用户，减少结算流程阻力，例如突出免邮、售后保障和支付便利性。
6. 后续可以将模型输出分数接入运营看板，用于每日高购买概率 session 排序。

## 15. 输出图表

主要图表输出在 `outputs/figures/` 目录下：

- `lightgbm_test_roc_curve.png`
- `lightgbm_test_pr_curve.png`
- `lightgbm_test_lift.png`
- `lightgbm_feature_importance_top20.png`
- `model_v2_comparison_test_auc.png`
- `model_v2_comparison_test_pr_auc.png`
- `model_v2_comparison_top10_lift.png`

## 16. 运行方式

建议按以下顺序运行核心脚本：

{code_fence}bash
python Notebook/11_Export_All_Months_Parquet.py
python Notebook/12_Multi_Month_KPI_Trend.py
python Notebook/13_Cohort_Retention_Analysis.py
python Notebook/14_Build_Model_Dataset_DuckDB.py
python Notebook/15_Train_LightGBM_Time_Split.py
python Notebook/16_Model_Evaluation_And_Interpretation.py
python Notebook/17_Generate_Project_Summary.py
{code_fence}

## 17. 注意事项

原始 CSV、Parquet 分区数据和模型文件体积较大，不建议上传到 GitHub。建议只上传：

- 源代码
- README
- 项目总结
- 小体量结果表
- 关键图表
- requirements.txt
- .gitignore

## 18. 项目价值

本项目完整覆盖了数据分析项目的核心链路：

1. 大规模数据清洗与列式存储。
2. 多月经营指标分析。
3. 用户留存与复购分析。
4. 防泄漏 session 级特征工程。
5. 时间切分机器学习建模。
6. baseline 对比与 ablation study。
7. 模型解释、错误分析和业务建议。

相比普通电商 EDA 项目，本项目更接近真实业务中的数据分析和数据挖掘流程。
"""

readme_path.write_text(readme_text, encoding="utf-8")
print("\nREADME.md 已生成：")
print(readme_path)

# =====================================================
# 7. 生成 PROJECT_SUMMARY.md
# =====================================================

project_summary_text = f"""# 项目总结：多月电商用户行为分析与购买转化预测

生成时间：{generated_time}

## 一、项目背景

本项目基于多品类电商平台用户行为日志，围绕浏览、加购和购买行为，完成大规模行为数据处理、经营指标分析、用户留存分析和购买转化预测建模。

项目处理的数据覆盖 2019-10 至 2020-04，共 {fmt_int(total_month_count)} 个月、{fmt_int(total_day_count)} 天，包含 {fmt_int(total_event_count)} 条有效行为记录、{fmt_int(total_user_count)} 名用户和 {fmt_int(total_session_count)} 个 session。

## 二、数据工程处理

原始数据为多个大体量 CSV 文件，不适合反复用 pandas 直接读取。因此，项目使用 DuckDB 对原始 CSV 进行清洗，并按月份和日期导出为 Parquet 分区数据。

这一处理将项目从普通小样本 EDA 提升为可处理大规模行为日志的数据分析流程。

## 三、多月经营指标分析

项目构建月度经营指标体系，包括浏览量、加购量、购买量、销售额、活跃用户数、购买用户数、session 数、购买 session 数、购买/浏览比例和平均购买价格等。

核心发现：

- 销售额最高月份为 {best_revenue_month}，销售额约 {fmt_number(best_revenue, 2)}。
- 购买次数最高月份购买事件数约 {fmt_int(best_purchase_count)}。
- 最高购买 / 浏览比例约为 {fmt_percent(best_purchase_view_ratio)}。

## 四、Cohort 留存与复购分析

项目基于用户首次购买月份构建 cohort 留存矩阵，分析不同首购月份用户在后续月份的复购表现。

核心发现：

- {best_cohort_month} cohort 次月留存率最高，达到 {fmt_percent(best_month_1_retention)}。
- {largest_cohort_month} cohort 首购用户规模最大，达到 {fmt_int(largest_cohort_user_count)} 人。
- {best_repeat_month} 的复购用户占比最高，达到 {fmt_percent(best_repeat_ratio)}。

该部分说明用户规模和用户质量并不完全一致，单纯追求新用户数量并不一定带来更高的后续留存。

## 五、购买预测建模

项目将行为日志聚合为 session 级样本，以 session 是否产生购买作为预测目标，构建 LightGBM 二分类模型。

建模采用时间切分：

- 训练集：2019-10 至 2020-02
- 验证集：2020-03
- 测试集：2020-04

新版建模数据包含 {fmt_int(train_row_count)} 条训练样本、{fmt_int(valid_row_count)} 条验证样本和 {fmt_int(test_row_count)} 条测试样本。

## 六、防泄漏处理

为提高建模可信度，项目进行了两类防泄漏处理：

1. 对购买 session，只使用首次 purchase 之前的 view/cart 行为构造特征。
2. 用户历史特征只使用 session 所在月份之前的数据。

这使模型更接近真实业务中“根据当前和历史行为预测未来购买”的场景。

## 七、最终模型表现

最终选择 LightGBM full model。该模型在 2020-04 测试集取得：

- ROC-AUC：{fmt_float(test_auc, 4)}
- PR-AUC：{fmt_float(test_pr_auc, 4)}
- Accuracy：{fmt_float(test_accuracy, 4)}
- Precision：{fmt_float(test_precision, 4)}
- Recall：{fmt_float(test_recall, 4)}
- F1-score：{fmt_float(test_f1, 4)}

## 八、模型对比与 Ablation Study

项目比较了 has_cart 规则模型、Logistic Regression 和多个 LightGBM 特征组模型。

- has_cart 规则 baseline 的 Top 10% Lift 为 {fmt_float(baseline_top10_lift, 2)}。
- Logistic Regression 的 Top 10% Lift 为 {fmt_float(logistic_top10_lift, 2)}。
- LightGBM full model 的 Top 10% Lift 为 {fmt_float(full_model_top10_lift, 2)}。

从 ablation study 看，仅使用 session 行为特征已经有较强表现；加入用户历史特征后，PR-AUC 从 {fmt_float(behavior_only_pr_auc, 4)} 提升到 {fmt_float(full_model_pr_auc, 4)}，说明用户历史行为具有增量价值。

## 九、Lift 分析

测试集整体购买率为 {fmt_percent(overall_purchase_rate)}。模型预测概率最高的前 10% session 中，真实购买率达到 {fmt_percent(top10_purchase_rate)}，Lift 为 {fmt_float(top10_lift, 2)} 倍，并覆盖 {fmt_percent(top10_captured_purchase_ratio)} 的真实购买 session。

这说明模型更适合用于营销资源排序，例如优惠券、客服触达、推荐位和购物车召回优先级分配。

## 十、特征解释

特征组重要性显示，最重要的特征组是 {top_feature_group_cn}，重要性占比约 {fmt_percent(top_feature_group_ratio)}。

单个最重要特征是 {top_feature}，说明首次购买前的加购次数是判断购买意向的核心信号。

## 十一、错误分析

测试集中：

- TP：{fmt_int(tp_count)}
- TN：{fmt_int(tn_count)}
- FP：{fmt_int(fp_count)}
- FN：{fmt_int(fn_count)}

False Positive 通常代表有明显加购或深度浏览行为但最终未购买的 session，可能受到价格、库存、物流或支付流程影响。False Negative 则可能代表快速决策或老用户直接购买场景。

## 十二、业务建议

1. 将模型分数用于营销资源排序，而不是简单二分类。
2. 对 Top 10% 高分 session 优先进行优惠券、限时折扣、库存提醒或客服触达。
3. 对加购但未购买用户进行购物车召回。
4. 对历史活跃和历史购买强的用户加强会员运营。
5. 对高购物车金额用户优化结算流程。
6. 后续可将模型分数接入运营看板，形成每日高购买概率 session 列表。

## 十三、项目价值

本项目覆盖了从数据工程、指标分析、用户生命周期分析到机器学习建模和业务解释的完整链路。相比普通电商 EDA 项目，本项目具备更强的数据规模、工程结构、建模严谨性和业务解释能力。
"""

summary_path.write_text(project_summary_text, encoding="utf-8")
print("\nPROJECT_SUMMARY.md 已生成：")
print(summary_path)

# =====================================================
# 8. 生成 RESUME_DESCRIPTION.md
# =====================================================

resume_text = f"""# 简历项目描述：多月电商用户行为分析与购买转化预测

## 项目名称

多月电商用户行为分析与购买转化预测

## 一句话简介

基于 Kaggle 多品类电商用户行为日志，使用 Python、DuckDB、Parquet、pandas 和 LightGBM 构建大规模用户行为分析与 session 级购买转化预测流程，覆盖数据工程、经营指标分析、Cohort 留存分析、机器学习建模和业务解释。

## 简历 Bullet：数据分析岗版本

- 使用 DuckDB 处理 2019-10 至 2020-04 共 {fmt_int(total_event_count)} 条有效电商用户行为日志，并将原始 CSV 清洗转换为按月份和日期分区的 Parquet 数据，提升大规模行为数据读取和聚合效率。
- 构建月度经营指标体系，统计浏览、加购、购买、销售额、购买用户数、购买/浏览比例和 session 转化情况，识别 {best_revenue_month} 为销售额与购买转化表现突出的月份。
- 基于用户首购月份构建 Cohort 留存矩阵，发现 {best_cohort_month} cohort 次月留存率最高，达到 {fmt_percent(best_month_1_retention)}；同时 {best_repeat_month} 复购用户占比达到 {fmt_percent(best_repeat_ratio)}。
- 将行为日志聚合为 session 级建模样本，构造 37 个防泄漏行为、价格、时间和用户历史特征，采用 2019-10 至 2020-02 训练、2020-03 验证、2020-04 测试的时间切分方式评估模型泛化能力。
- 使用 LightGBM 构建购买预测模型，在 2020-04 测试集取得 ROC-AUC {fmt_float(test_auc, 4)}、PR-AUC {fmt_float(test_pr_auc, 4)}；模型 Top 10% 高分 session 购买率达到 {fmt_percent(top10_purchase_rate)}，相较整体购买率提升 {fmt_float(top10_lift, 2)} 倍，可用于精准营销资源排序。

## 简历 Bullet：数据挖掘 / 建模岗版本

- 面向 {fmt_int(total_event_count)} 条多月电商行为日志，设计 DuckDB + Parquet 的大规模数据处理流程，将原始 CSV 转换为按月份 / 日期分区的列式存储，并基于 SQL 完成大规模聚合特征工程。
- 将用户行为日志聚合为 session 级样本，以 session 是否购买作为二分类目标；构造浏览深度、加购行为、价格区间、session 持续时间、用户历史活跃度和历史购买行为等 37 个特征。
- 设计防泄漏特征工程：购买 session 仅使用首次 purchase 前的 view/cart 行为，用户历史特征仅使用 session 所在月份之前的数据，避免时间穿越。
- 对比 has_cart 规则 baseline、Logistic Regression 和多组 LightGBM ablation 模型，发现 LightGBM full model 在测试集上取得最佳综合表现，ROC-AUC {fmt_float(test_auc, 4)}、PR-AUC {fmt_float(test_pr_auc, 4)}、F1 {fmt_float(test_f1, 4)}。
- 通过 Lift 分析验证模型业务价值：Top 10% 高分 session 购买率达到 {fmt_percent(top10_purchase_rate)}，Lift 为 {fmt_float(top10_lift, 2)}，覆盖 {fmt_percent(top10_captured_purchase_ratio)} 的真实购买 session，可用于优惠券、客服触达和推荐位的优先级排序。
- 通过特征组重要性和错误分析解释模型结果，发现当前 session 加购行为贡献约 {fmt_percent(top_feature_group_ratio)} 的特征重要性，是购买预测的核心信号。

## 面试讲解稿

这个项目不是单纯做电商 EDA，而是从大规模行为日志出发，构建了完整的数据分析和预测建模流程。

首先，我处理的是 2019-10 到 2020-04 共 7 个月的电商用户行为日志，包含约 {fmt_int(total_event_count)} 条有效行为记录和 {fmt_int(total_session_count)} 个 session。因为原始 CSV 文件较大，我没有反复用 pandas 读取，而是使用 DuckDB 对数据进行清洗，并导出为按月份和日期分区的 Parquet 文件，这样后续分析和建模都可以直接基于列式存储进行。

第二步，我构建了月度 KPI 指标体系，分析浏览、加购、购买、销售额、购买用户数和购买/浏览比例等指标。之后，我基于用户首次购买月份构建 Cohort 留存矩阵，分析不同首购月份用户的后续复购表现，发现用户规模和用户质量并不完全一致。

第三步是建模。我将行为日志聚合为 session 级样本，以 session 是否发生购买作为预测目标。为了更接近真实业务场景，我采用时间切分：2019-10 到 2020-02 作为训练集，2020-03 作为验证集，2020-04 作为测试集。特征工程上，我特别做了防泄漏处理：对于购买 session，只使用首次 purchase 之前的 view/cart 行为；用户历史特征也只使用 session 所在月份之前的数据。

模型方面，我比较了 has_cart 规则 baseline、Logistic Regression 和多个 LightGBM 特征组模型。最终 LightGBM full model 在测试集上取得 ROC-AUC {fmt_float(test_auc, 4)}、PR-AUC {fmt_float(test_pr_auc, 4)}。从业务角度看，模型预测概率最高的前 10% session 购买率达到 {fmt_percent(top10_purchase_rate)}，是整体购买率的 {fmt_float(top10_lift, 2)} 倍，并覆盖 {fmt_percent(top10_captured_purchase_ratio)} 的真实购买 session，因此它很适合用于营销资源排序，比如优惠券、客服触达和推荐位分配。

最后，我通过特征重要性和错误分析解释模型。结果显示，首次购买前的加购次数是最重要的特征，当前 session 加购行为是最重要的特征组。这说明在电商购买预测中，加购行为是最核心的购买意向信号，而用户历史活跃和历史购买行为提供了额外增量价值。

## 项目关键词

Python、DuckDB、Parquet、pandas、LightGBM、scikit-learn、Cohort Analysis、Feature Engineering、Binary Classification、Lift Analysis、Ablation Study、E-commerce Analytics
"""

resume_path.write_text(resume_text, encoding="utf-8")
print("\nRESUME_DESCRIPTION.md 已生成：")
print(resume_path)

# =====================================================
# 9. 生成 requirements.txt
# =====================================================

requirements_text = """duckdb
pandas
numpy
matplotlib
scikit-learn
lightgbm
joblib
openpyxl
pyarrow
"""

requirements_path.write_text(requirements_text, encoding="utf-8")
print("\nrequirements.txt 已生成：")
print(requirements_path)

# =====================================================
# 10. 生成 .gitignore
# =====================================================

gitignore_text = """# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environment
.venv/
venv/
env/

# IDE
.idea/
.vscode/

# Raw and processed large data
data/raw/
data/processed/
data/modeling/*.parquet
Data/
*.parquet
*.csv.gz
*.zip

# DuckDB temp and database files
duckdb_temp/
*.duckdb
*.db

# Model files
models/*.pkl
models/*.joblib

# Large prediction output
outputs/tables/lightgbm_test_prediction_top50000.csv

# System files
.DS_Store
Thumbs.db

# Logs
*.log
"""

gitignore_path.write_text(gitignore_text, encoding="utf-8")
print("\n.gitignore 已生成：")
print(gitignore_path)

# =====================================================
# 11. 最终输出检查
# =====================================================

print("\n第 17 步 V2 生成文件：")
print(readme_path)
print(summary_path)
print(resume_path)
print(requirements_path)
print(gitignore_path)

print("\n核心指标确认：")
print(f"测试集 ROC-AUC: {fmt_float(test_auc, 4)}")
print(f"测试集 PR-AUC: {fmt_float(test_pr_auc, 4)}")
print(f"测试集 F1: {fmt_float(test_f1, 4)}")
print(f"Top 10% 购买率: {fmt_percent(top10_purchase_rate)}")
print(f"Top 10% Lift: {fmt_float(top10_lift, 2)}")
print(f"Top 10% 捕获真实购买比例: {fmt_percent(top10_captured_purchase_ratio)}")

print("\n第 17 步 V2：README、项目总结、简历描述、requirements 和 .gitignore 生成完成。")