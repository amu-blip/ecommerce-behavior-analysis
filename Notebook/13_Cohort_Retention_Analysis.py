from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# =====================================================
# 1. 定位项目路径
# =====================================================

current_file = Path(__file__).resolve()
project_dir = current_file.parents[1]

processed_root = project_dir / "Data" / "Processed"
figures_dir = project_dir / "outputs" / "figures"
tables_dir = project_dir / "outputs" / "tables"

figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

print("项目根目录：")
print(project_dir)

print("\nParquet 数据目录：")
print(processed_root)

if not processed_root.exists():
    raise FileNotFoundError(f"找不到 Parquet 数据目录：{processed_root}")

# =====================================================
# 2. 只读取指定月份目录，避免误读旧 Parquet
# =====================================================

month_folders = [
    "2019_10",
    "2019_11",
    "2019_12",
    "2020_01",
    "2020_02",
    "2020_03",
    "2020_04"
]

parquet_files = []

for month in month_folders:
    month_dir = processed_root / month

    if not month_dir.exists():
        print(f"警告：月份目录不存在，跳过：{month_dir}")
        continue

    files = sorted(month_dir.glob("*/*.parquet"))
    parquet_files.extend(files)

print("\n检测到 Parquet 文件数量：")
print(len(parquet_files))

if len(parquet_files) == 0:
    raise FileNotFoundError("没有找到任何 Parquet 文件，请检查 Data/Processed 目录结构。")

parquet_file_list = [
    str(file).replace("\\", "/")
    for file in parquet_files
]

print("\n前 10 个将要读取的 Parquet 文件：")
for file in parquet_file_list[:10]:
    print(file)

# =====================================================
# 3. DuckDB 读取 Parquet 文件
# =====================================================
# 说明：
# DuckDB 的 CREATE VIEW 不能直接使用 read_parquet(?) 参数绑定
# 所以这里把 parquet 文件路径列表拼接成 SQL list

parquet_file_sql_list = "[" + ", ".join(
    "'" + file.replace("'", "''") + "'"
    for file in parquet_file_list
) + "]"

con = duckdb.connect()

con.execute(f"""
CREATE OR REPLACE VIEW ecommerce_all AS
SELECT
    event_ts,
    event_date,
    CAST(date_trunc('month', event_ts) AS DATE) AS purchase_month_date,
    strftime(event_ts, '%Y_%m') AS event_month,
    event_type,
    product_id,
    category_id,
    category_code,
    main_category,
    brand,
    CAST(price AS DOUBLE) AS price,
    user_id,
    user_session,
    hour,
    weekday
FROM read_parquet({parquet_file_sql_list})
WHERE event_ts IS NOT NULL
  AND price > 0
""")

print("\nDuckDB 视图 ecommerce_all 创建完成。")

# =====================================================
# 4. 全量数据检查
# =====================================================

data_check_sql = """
SELECT
    MIN(event_ts) AS min_time,
    MAX(event_ts) AS max_time,
    COUNT(*) AS row_count,
    COUNT(DISTINCT event_month) AS month_count,
    COUNT(DISTINCT event_date) AS date_count,
    COUNT(DISTINCT user_id) AS user_count
FROM ecommerce_all
"""

data_check = con.execute(data_check_sql).df()

print("\n全量数据检查：")
print(data_check)

# =====================================================
# 5. 购买数据检查
# =====================================================

purchase_check_sql = """
SELECT
    MIN(event_ts) AS min_purchase_time,
    MAX(event_ts) AS max_purchase_time,
    COUNT(*) AS purchase_event_count,
    COUNT(DISTINCT user_id) AS purchase_user_count,
    COUNT(DISTINCT purchase_month_date) AS purchase_month_count
FROM ecommerce_all
WHERE event_type = 'purchase'
"""

purchase_check = con.execute(purchase_check_sql).df()

print("\n购买数据检查：")
print(purchase_check)

# =====================================================
# 6. 构建 Cohort 留存数据
# =====================================================
# cohort_month：用户第一次购买所在月份
# activity_month：用户发生购买行为的月份
# month_index：距离首购月份过去几个月
# month_index = 0 表示首购当月，理论上为 100%

cohort_sql = """
WITH purchase_monthly AS (
    SELECT DISTINCT
        user_id,
        purchase_month_date AS activity_month
    FROM ecommerce_all
    WHERE event_type = 'purchase'
),

first_purchase AS (
    SELECT
        user_id,
        MIN(activity_month) AS cohort_month
    FROM purchase_monthly
    GROUP BY user_id
),

cohort_activity AS (
    SELECT
        f.cohort_month,
        p.activity_month,
        DATE_DIFF('month', f.cohort_month, p.activity_month) AS month_index,
        COUNT(DISTINCT p.user_id) AS active_users
    FROM first_purchase f
    JOIN purchase_monthly p
    ON f.user_id = p.user_id
    GROUP BY
        f.cohort_month,
        p.activity_month,
        DATE_DIFF('month', f.cohort_month, p.activity_month)
),

cohort_size AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT user_id) AS cohort_user_count
    FROM first_purchase
    GROUP BY cohort_month
)

SELECT
    strftime(a.cohort_month, '%Y_%m') AS cohort_month,
    strftime(a.activity_month, '%Y_%m') AS activity_month,
    a.month_index,
    s.cohort_user_count,
    a.active_users,
    a.active_users * 1.0 / s.cohort_user_count AS retention_rate
FROM cohort_activity a
JOIN cohort_size s
ON a.cohort_month = s.cohort_month
ORDER BY
    a.cohort_month,
    a.month_index
"""

cohort_result = con.execute(cohort_sql).df()

print("\nCohort 留存明细：")
print(cohort_result)

cohort_detail_csv = tables_dir / "cohort_retention_detail.csv"
cohort_detail_xlsx = tables_dir / "cohort_retention_detail.xlsx"

cohort_result.to_csv(cohort_detail_csv, index=False, encoding="utf-8-sig")
cohort_result.to_excel(cohort_detail_xlsx, index=False)

print("\nCohort 留存明细表已保存：")
print(cohort_detail_csv)
print(cohort_detail_xlsx)

# =====================================================
# 7. 生成留存率矩阵和留存人数矩阵
# =====================================================

retention_rate_matrix = cohort_result.pivot(
    index="cohort_month",
    columns="month_index",
    values="retention_rate"
)

retention_count_matrix = cohort_result.pivot(
    index="cohort_month",
    columns="month_index",
    values="active_users"
)

print("\nCohort 留存率矩阵：")
print(retention_rate_matrix)

print("\nCohort 留存人数矩阵：")
print(retention_count_matrix)

retention_rate_csv = tables_dir / "cohort_retention_rate_matrix.csv"
retention_rate_xlsx = tables_dir / "cohort_retention_rate_matrix.xlsx"

retention_count_csv = tables_dir / "cohort_retention_count_matrix.csv"
retention_count_xlsx = tables_dir / "cohort_retention_count_matrix.xlsx"

retention_rate_matrix.to_csv(retention_rate_csv, encoding="utf-8-sig")
retention_rate_matrix.to_excel(retention_rate_xlsx)

retention_count_matrix.to_csv(retention_count_csv, encoding="utf-8-sig")
retention_count_matrix.to_excel(retention_count_xlsx)

print("\nCohort 留存率矩阵已保存：")
print(retention_rate_csv)
print(retention_rate_xlsx)

print("\nCohort 留存人数矩阵已保存：")
print(retention_count_csv)
print(retention_count_xlsx)

# =====================================================
# 8. 每月新购用户与复购用户
# =====================================================

monthly_new_repeat_sql = """
WITH purchase_monthly AS (
    SELECT DISTINCT
        user_id,
        purchase_month_date AS activity_month
    FROM ecommerce_all
    WHERE event_type = 'purchase'
),

first_purchase AS (
    SELECT
        user_id,
        MIN(activity_month) AS cohort_month
    FROM purchase_monthly
    GROUP BY user_id
),

monthly_user_type AS (
    SELECT
        p.activity_month,
        p.user_id,
        f.cohort_month,
        CASE
            WHEN p.activity_month = f.cohort_month THEN 'new_purchase_user'
            ELSE 'repeat_purchase_user'
        END AS user_type
    FROM purchase_monthly p
    JOIN first_purchase f
    ON p.user_id = f.user_id
)

SELECT
    strftime(activity_month, '%Y_%m') AS event_month,

    COUNT(DISTINCT CASE
        WHEN user_type = 'new_purchase_user' THEN user_id
        ELSE NULL
    END) AS new_purchase_user_count,

    COUNT(DISTINCT CASE
        WHEN user_type = 'repeat_purchase_user' THEN user_id
        ELSE NULL
    END) AS repeat_purchase_user_count,

    COUNT(DISTINCT user_id) AS total_purchase_user_count,

    COUNT(DISTINCT CASE
        WHEN user_type = 'repeat_purchase_user' THEN user_id
        ELSE NULL
    END) * 1.0 / NULLIF(COUNT(DISTINCT user_id), 0) AS repeat_user_ratio

FROM monthly_user_type
GROUP BY activity_month
ORDER BY activity_month
"""

monthly_new_repeat = con.execute(monthly_new_repeat_sql).df()

print("\n每月新购用户与复购用户：")
print(monthly_new_repeat)

new_repeat_csv = tables_dir / "monthly_new_repeat_purchase_users.csv"
new_repeat_xlsx = tables_dir / "monthly_new_repeat_purchase_users.xlsx"

monthly_new_repeat.to_csv(new_repeat_csv, index=False, encoding="utf-8-sig")
monthly_new_repeat.to_excel(new_repeat_xlsx, index=False)

print("\n每月新购/复购用户表已保存：")
print(new_repeat_csv)
print(new_repeat_xlsx)

# =====================================================
# 9. Cohort 概览表
# =====================================================

cohort_summary = cohort_result[cohort_result["month_index"] == 0][[
    "cohort_month",
    "cohort_user_count"
]].copy()

month_1_retention = cohort_result[cohort_result["month_index"] == 1][[
    "cohort_month",
    "active_users",
    "retention_rate"
]].copy()

month_1_retention = month_1_retention.rename(columns={
    "active_users": "month_1_retained_users",
    "retention_rate": "month_1_retention_rate"
})

cohort_summary = pd.merge(
    cohort_summary,
    month_1_retention,
    on="cohort_month",
    how="left"
)

cohort_summary["month_1_retained_users"] = cohort_summary["month_1_retained_users"].fillna(0).astype(int)
cohort_summary["month_1_retention_rate"] = cohort_summary["month_1_retention_rate"].fillna(0)

cohort_summary = cohort_summary.sort_values("cohort_month")

print("\nCohort 概览表：")
print(cohort_summary)

cohort_summary_csv = tables_dir / "cohort_summary.csv"
cohort_summary_xlsx = tables_dir / "cohort_summary.xlsx"

cohort_summary.to_csv(cohort_summary_csv, index=False, encoding="utf-8-sig")
cohort_summary.to_excel(cohort_summary_xlsx, index=False)

print("\nCohort 概览表已保存：")
print(cohort_summary_csv)
print(cohort_summary_xlsx)

# =====================================================
# 10. 绘图：Cohort 留存率热力图
# =====================================================

plot_matrix = retention_rate_matrix.copy()
data_for_plot = plot_matrix.values.astype(float)

plt.figure(figsize=(10, 6))

plt.imshow(data_for_plot, aspect="auto")
plt.colorbar(label="Retention Rate")

plt.title("Cohort Retention Rate Matrix")
plt.xlabel("Months Since First Purchase")
plt.ylabel("Cohort Month")

plt.xticks(
    ticks=np.arange(len(plot_matrix.columns)),
    labels=[str(col) for col in plot_matrix.columns]
)

plt.yticks(
    ticks=np.arange(len(plot_matrix.index)),
    labels=plot_matrix.index
)

for i in range(data_for_plot.shape[0]):
    for j in range(data_for_plot.shape[1]):
        value = data_for_plot[i, j]
        if not np.isnan(value):
            plt.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=8)

plt.tight_layout()

heatmap_fig = figures_dir / "cohort_retention_heatmap.png"
plt.savefig(heatmap_fig, dpi=300)
plt.show()

print("\nCohort 留存率热力图已保存：")
print(heatmap_fig)

# =====================================================
# 11. 绘图：每月新购与复购用户数
# =====================================================

x = np.arange(len(monthly_new_repeat["event_month"]))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(
    x - width / 2,
    monthly_new_repeat["new_purchase_user_count"],
    width,
    label="New Purchase Users"
)

plt.bar(
    x + width / 2,
    monthly_new_repeat["repeat_purchase_user_count"],
    width,
    label="Repeat Purchase Users"
)

plt.title("Monthly New vs Repeat Purchase Users")
plt.xlabel("Month")
plt.ylabel("User Count")
plt.xticks(x, monthly_new_repeat["event_month"], rotation=30)
plt.legend()

plt.tight_layout()

new_repeat_fig = figures_dir / "monthly_new_repeat_purchase_users.png"
plt.savefig(new_repeat_fig, dpi=300)
plt.show()

print("\n每月新购与复购用户图已保存：")
print(new_repeat_fig)

# =====================================================
# 12. 绘图：复购用户占比趋势
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_new_repeat["event_month"],
    monthly_new_repeat["repeat_user_ratio"] * 100,
    marker="o"
)

plt.title("Monthly Repeat Purchase User Ratio")
plt.xlabel("Month")
plt.ylabel("Repeat User Ratio (%)")
plt.xticks(rotation=30)

for i, value in enumerate(monthly_new_repeat["repeat_user_ratio"] * 100):
    plt.text(i, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

repeat_ratio_fig = figures_dir / "monthly_repeat_user_ratio.png"
plt.savefig(repeat_ratio_fig, dpi=300)
plt.show()

print("\n月度复购用户占比趋势图已保存：")
print(repeat_ratio_fig)

# =====================================================
# 13. 输出关键结论辅助信息
# =====================================================

print("\n首购用户规模最大的 cohort：")
print(cohort_summary.sort_values("cohort_user_count", ascending=False).head(3))

print("\n次月留存率最高的 cohort：")
print(cohort_summary.sort_values("month_1_retention_rate", ascending=False).head(3))

print("\n复购用户占比最高的月份：")
print(monthly_new_repeat.sort_values("repeat_user_ratio", ascending=False).head(3))

print("\nCohort 留存分析完成。")