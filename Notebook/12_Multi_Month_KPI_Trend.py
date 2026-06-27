from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

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
# 2. 读取所有月份的 Parquet 文件
# =====================================================

# =====================================================
# 2. 只读取指定月份目录下的 Parquet 文件
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

    files = list(month_dir.glob("*/*.parquet"))
    parquet_files.extend(files)

print("\n检测到 Parquet 文件数量：")
print(len(parquet_files))

if len(parquet_files) == 0:
    raise FileNotFoundError("没有找到任何 Parquet 文件，请检查 Data/Processed 目录结构。")

# DuckDB 读取多个 parquet 文件时，使用文件路径列表
parquet_file_list = [
    str(file).replace("\\", "/")
    for file in parquet_files
]

print("\n前 10 个将要读取的 Parquet 文件：")
for file in parquet_file_list[:10]:
    print(file)

# =====================================================
# 3. 连接 DuckDB
# =====================================================

con = duckdb.connect()

con.execute("""
CREATE OR REPLACE VIEW ecommerce_all AS
SELECT
    event_ts,
    event_date,
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
FROM read_parquet(?)
WHERE event_ts IS NOT NULL
  AND price > 0
""", [parquet_file_list])

print("\nDuckDB 视图 ecommerce_all 创建完成。")

# =====================================================
# 4. 全量数据检查
# =====================================================

check_sql = """
SELECT
    MIN(event_ts) AS min_time,
    MAX(event_ts) AS max_time,
    COUNT(*) AS row_count,
    COUNT(DISTINCT event_month) AS month_count,
    COUNT(DISTINCT event_date) AS date_count,
    COUNT(DISTINCT user_id) AS user_count,
    COUNT(DISTINCT user_session) AS session_count
FROM ecommerce_all
"""

check_result = con.execute(check_sql).df()

print("\n全量 Parquet 数据检查：")
print(check_result)

month_check_sql = """
SELECT
    event_month,
    MIN(event_date) AS min_date,
    MAX(event_date) AS max_date,
    COUNT(*) AS row_count,
    COUNT(DISTINCT event_date) AS date_count
FROM ecommerce_all
GROUP BY event_month
ORDER BY event_month
"""

month_check = con.execute(month_check_sql).df()

print("\n各月份数据检查：")
print(month_check)

month_check_csv = tables_dir / "multi_month_data_check.csv"
month_check_xlsx = tables_dir / "multi_month_data_check.xlsx"

month_check.to_csv(month_check_csv, index=False, encoding="utf-8-sig")
month_check.to_excel(month_check_xlsx, index=False)

print("\n各月份数据检查表已保存：")
print(month_check_csv)
print(month_check_xlsx)

# =====================================================
# 5. 月度 KPI 汇总
# =====================================================

monthly_kpi_sql = """
WITH monthly AS (
    SELECT
        event_month,

        COUNT(*) AS total_events,

        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
        SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_count,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,

        SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS revenue,

        COUNT(DISTINCT user_id) AS active_user_count,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id ELSE NULL END) AS purchase_user_count,

        COUNT(DISTINCT CASE WHEN user_session != 'unknown' THEN user_session ELSE NULL END) AS session_count,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' AND user_session != 'unknown' THEN user_session ELSE NULL END) AS purchase_session_count

    FROM ecommerce_all
    GROUP BY event_month
)
SELECT
    event_month,
    total_events,
    view_count,
    cart_count,
    purchase_count,
    revenue,
    active_user_count,
    purchase_user_count,
    session_count,
    purchase_session_count,

    cart_count * 1.0 / NULLIF(view_count, 0) AS cart_view_ratio,
    purchase_count * 1.0 / NULLIF(view_count, 0) AS purchase_view_ratio,
    purchase_count * 1.0 / NULLIF(cart_count, 0) AS purchase_cart_ratio,

    revenue * 1.0 / NULLIF(purchase_count, 0) AS avg_purchase_price,
    revenue * 1.0 / NULLIF(purchase_user_count, 0) AS revenue_per_purchase_user,
    purchase_user_count * 1.0 / NULLIF(active_user_count, 0) AS purchase_user_ratio,
    purchase_session_count * 1.0 / NULLIF(session_count, 0) AS purchase_session_ratio

FROM monthly
ORDER BY event_month
"""

monthly_kpi = con.execute(monthly_kpi_sql).df()

print("\n月度 KPI 汇总：")
print(monthly_kpi)

monthly_kpi_csv = tables_dir / "monthly_kpi_summary.csv"
monthly_kpi_xlsx = tables_dir / "monthly_kpi_summary.xlsx"

monthly_kpi.to_csv(monthly_kpi_csv, index=False, encoding="utf-8-sig")
monthly_kpi.to_excel(monthly_kpi_xlsx, index=False)

print("\n月度 KPI 汇总表已保存：")
print(monthly_kpi_csv)
print(monthly_kpi_xlsx)

# =====================================================
# 6. 月度行为结构占比
# =====================================================

monthly_behavior_ratio = monthly_kpi[[
    "event_month",
    "view_count",
    "cart_count",
    "purchase_count"
]].copy()

monthly_behavior_ratio["view_ratio"] = monthly_behavior_ratio["view_count"] / (
    monthly_behavior_ratio["view_count"] +
    monthly_behavior_ratio["cart_count"] +
    monthly_behavior_ratio["purchase_count"]
)

monthly_behavior_ratio["cart_ratio"] = monthly_behavior_ratio["cart_count"] / (
    monthly_behavior_ratio["view_count"] +
    monthly_behavior_ratio["cart_count"] +
    monthly_behavior_ratio["purchase_count"]
)

monthly_behavior_ratio["purchase_ratio"] = monthly_behavior_ratio["purchase_count"] / (
    monthly_behavior_ratio["view_count"] +
    monthly_behavior_ratio["cart_count"] +
    monthly_behavior_ratio["purchase_count"]
)

print("\n月度行为结构占比：")
print(monthly_behavior_ratio)

behavior_ratio_csv = tables_dir / "monthly_behavior_ratio.csv"
behavior_ratio_xlsx = tables_dir / "monthly_behavior_ratio.xlsx"

monthly_behavior_ratio.to_csv(behavior_ratio_csv, index=False, encoding="utf-8-sig")
monthly_behavior_ratio.to_excel(behavior_ratio_xlsx, index=False)

print("\n月度行为结构占比表已保存：")
print(behavior_ratio_csv)
print(behavior_ratio_xlsx)

# =====================================================
# 7. 绘图：月度销售额趋势
# =====================================================

plt.figure(figsize=(10, 5))
plt.plot(monthly_kpi["event_month"], monthly_kpi["revenue"], marker="o")

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=30)

for i, value in enumerate(monthly_kpi["revenue"]):
    plt.text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

monthly_revenue_fig = figures_dir / "monthly_revenue_trend.png"
plt.savefig(monthly_revenue_fig, dpi=300)
plt.show()

print("\n月度销售额趋势图已保存：")
print(monthly_revenue_fig)

# =====================================================
# 8. 绘图：月度购买次数趋势
# =====================================================

plt.figure(figsize=(10, 5))
plt.plot(monthly_kpi["event_month"], monthly_kpi["purchase_count"], marker="o")

plt.title("Monthly Purchase Count Trend")
plt.xlabel("Month")
plt.ylabel("Purchase Count")
plt.xticks(rotation=30)

for i, value in enumerate(monthly_kpi["purchase_count"]):
    plt.text(i, value, f"{int(value):,}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

monthly_purchase_fig = figures_dir / "monthly_purchase_count_trend.png"
plt.savefig(monthly_purchase_fig, dpi=300)
plt.show()

print("\n月度购买次数趋势图已保存：")
print(monthly_purchase_fig)

# =====================================================
# 9. 绘图：月度购买用户数趋势
# =====================================================

plt.figure(figsize=(10, 5))
plt.plot(monthly_kpi["event_month"], monthly_kpi["purchase_user_count"], marker="o")

plt.title("Monthly Purchase User Count Trend")
plt.xlabel("Month")
plt.ylabel("Purchase User Count")
plt.xticks(rotation=30)

for i, value in enumerate(monthly_kpi["purchase_user_count"]):
    plt.text(i, value, f"{int(value):,}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

monthly_purchase_user_fig = figures_dir / "monthly_purchase_user_count_trend.png"
plt.savefig(monthly_purchase_user_fig, dpi=300)
plt.show()

print("\n月度购买用户数趋势图已保存：")
print(monthly_purchase_user_fig)

# =====================================================
# 10. 绘图：月度购买/浏览比例趋势
# =====================================================

plt.figure(figsize=(10, 5))
plt.plot(
    monthly_kpi["event_month"],
    monthly_kpi["purchase_view_ratio"] * 100,
    marker="o"
)

plt.title("Monthly Purchase/View Ratio Trend")
plt.xlabel("Month")
plt.ylabel("Purchase/View Ratio (%)")
plt.xticks(rotation=30)

for i, value in enumerate(monthly_kpi["purchase_view_ratio"] * 100):
    plt.text(i, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

monthly_conversion_fig = figures_dir / "monthly_purchase_view_ratio_trend.png"
plt.savefig(monthly_conversion_fig, dpi=300)
plt.show()

print("\n月度购买/浏览比例趋势图已保存：")
print(monthly_conversion_fig)

# =====================================================
# 11. 绘图：月度客单价趋势
# =====================================================

plt.figure(figsize=(10, 5))
plt.plot(
    monthly_kpi["event_month"],
    monthly_kpi["avg_purchase_price"],
    marker="o"
)

plt.title("Monthly Average Purchase Price Trend")
plt.xlabel("Month")
plt.ylabel("Average Purchase Price")
plt.xticks(rotation=30)

for i, value in enumerate(monthly_kpi["avg_purchase_price"]):
    plt.text(i, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

monthly_avg_price_fig = figures_dir / "monthly_avg_purchase_price_trend.png"
plt.savefig(monthly_avg_price_fig, dpi=300)
plt.show()

print("\n月度平均购买价格趋势图已保存：")
print(monthly_avg_price_fig)

# =====================================================
# 12. 输出关键结论辅助信息
# =====================================================

print("\n销售额最高的月份：")
print(monthly_kpi.sort_values("revenue", ascending=False).head(3))

print("\n购买次数最高的月份：")
print(monthly_kpi.sort_values("purchase_count", ascending=False).head(3))

print("\n购买/浏览比例最高的月份：")
print(monthly_kpi.sort_values("purchase_view_ratio", ascending=False).head(3))

print("\n购买用户数最高的月份：")
print(monthly_kpi.sort_values("purchase_user_count", ascending=False).head(3))

print("\n多月 KPI 趋势分析完成。")