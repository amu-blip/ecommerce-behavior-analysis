from pathlib import Path
import duckdb
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# =====================================================
# 1. 定位项目路径
# =====================================================

current_file = Path(__file__).resolve()
project_dir = current_file.parents[1]

processed_root = project_dir / "Data" / "Processed"
modeling_dir = project_dir / "data" / "modeling"
tables_dir = project_dir / "outputs" / "tables"

modeling_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

print("项目根目录：")
print(project_dir)

print("\nParquet 数据目录：")
print(processed_root)

print("\n建模数据输出目录：")
print(modeling_dir)

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

# DuckDB 的 CREATE VIEW 不能直接 read_parquet(?) 参数绑定
parquet_file_sql_list = "[" + ", ".join(
    "'" + file.replace("'", "''") + "'"
    for file in parquet_file_list
) + "]"

# =====================================================
# 3. 建模抽样参数
# =====================================================
# 说明：
# BP = basis point，100 / 10000 = 1%
#
# 本版比上一版提高抽样比例：
# 训练集约 2%
# 验证集约 3%
# 测试集约 3%
#
# 如果内存压力过大，可以改回：
# TRAIN_SAMPLE_RATE_BP = 100
# VALID_SAMPLE_RATE_BP = 200
# TEST_SAMPLE_RATE_BP = 200

TRAIN_SAMPLE_RATE_BP = 200
VALID_SAMPLE_RATE_BP = 300
TEST_SAMPLE_RATE_BP = 300

HASH_BASE = 10000

print("\n抽样参数：")
print(f"TRAIN_SAMPLE_RATE_BP = {TRAIN_SAMPLE_RATE_BP} / {HASH_BASE}")
print(f"VALID_SAMPLE_RATE_BP = {VALID_SAMPLE_RATE_BP} / {HASH_BASE}")
print(f"TEST_SAMPLE_RATE_BP = {TEST_SAMPLE_RATE_BP} / {HASH_BASE}")

# =====================================================
# 4. 连接 DuckDB
# =====================================================

con = duckdb.connect()

# 资源控制
# 16GB 内存建议 threads=4
# 32GB 内存可尝试 threads=6 或 8
con.execute("PRAGMA threads=4")
con.execute("PRAGMA preserve_insertion_order=false")

duckdb_temp_dir = project_dir / "duckdb_temp"
duckdb_temp_dir.mkdir(parents=True, exist_ok=True)

duckdb_temp_path = str(duckdb_temp_dir).replace("\\", "/")
con.execute(f"PRAGMA temp_directory='{duckdb_temp_path}'")

print("\nDuckDB 临时目录：")
print(duckdb_temp_dir)

# =====================================================
# 5. 创建全量事件视图
# =====================================================

con.execute(f"""
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
FROM read_parquet({parquet_file_sql_list})
WHERE event_ts IS NOT NULL
  AND price > 0
  AND user_session IS NOT NULL
  AND user_session != 'unknown'
""")

print("\nDuckDB 视图 ecommerce_all 创建完成。")

# =====================================================
# 6. 全量数据检查
# =====================================================

data_check_sql = """
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

data_check = con.execute(data_check_sql).df()

print("\n全量数据检查：")
print(data_check)

# =====================================================
# 7. 各月份 session 检查
# =====================================================

monthly_session_check_sql = """
SELECT
    event_month,
    COUNT(DISTINCT user_session) AS session_count,
    COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_session ELSE NULL END) AS purchase_session_count,
    COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_session ELSE NULL END) * 1.0
        / NULLIF(COUNT(DISTINCT user_session), 0) AS purchase_session_ratio
FROM ecommerce_all
GROUP BY event_month
ORDER BY event_month
"""

monthly_session_check = con.execute(monthly_session_check_sql).df()

print("\n各月份 Session 与购买 Session 检查：")
print(monthly_session_check)

monthly_session_check_csv = tables_dir / "model_v2_monthly_session_check.csv"
monthly_session_check_xlsx = tables_dir / "model_v2_monthly_session_check.xlsx"

monthly_session_check.to_csv(monthly_session_check_csv, index=False, encoding="utf-8-sig")
monthly_session_check.to_excel(monthly_session_check_xlsx, index=False)

print("\n各月份 Session 检查表已保存：")
print(monthly_session_check_csv)
print(monthly_session_check_xlsx)

# =====================================================
# 8. 抽取建模 session
# =====================================================
# 说明：
# 先抽 session，再聚合特征。
# 这样避免对全部 8961 万 session 做完整特征工程。
#
# 抽样使用 hash(user_session)，是确定性抽样。
# 同一个 session 的所有事件会一起进入样本。

sampled_sessions_sql = f"""
CREATE OR REPLACE TEMP TABLE sampled_sessions AS
SELECT DISTINCT
    user_session
FROM ecommerce_all
WHERE
    (
        event_month IN ('2019_10', '2019_11', '2019_12', '2020_01', '2020_02')
        AND hash(user_session) % {HASH_BASE} < {TRAIN_SAMPLE_RATE_BP}
    )
    OR
    (
        event_month = '2020_03'
        AND hash(user_session) % {HASH_BASE} < {VALID_SAMPLE_RATE_BP}
    )
    OR
    (
        event_month = '2020_04'
        AND hash(user_session) % {HASH_BASE} < {TEST_SAMPLE_RATE_BP}
    )
"""

print("\n开始抽取建模 session...")

con.execute(sampled_sessions_sql)

sampled_session_count = con.execute("""
SELECT COUNT(*) AS sampled_session_count
FROM sampled_sessions
""").df()

print("\n抽样 session 数量：")
print(sampled_session_count)

# =====================================================
# 9. 构建防泄漏版 session 级行为特征
# =====================================================
# 核心修改：
# 对购买 session，只使用首次 purchase 之前的 view/cart 行为构造特征。
#
# 例如：
# view -> cart -> purchase -> view
#
# 旧版可能统计 purchase 后的 view/cart；
# 新版只统计 purchase 前的 view/cart。
#
# 对非购买 session，则统计整个 session 中的 view/cart。
#
# target_purchase 仍然表示 session 是否发生 purchase。

session_feature_sql = """
CREATE OR REPLACE TEMP TABLE sampled_session_features AS
WITH sampled_events AS (
    SELECT
        e.*
    FROM ecommerce_all e
    JOIN sampled_sessions s
    ON e.user_session = s.user_session
),

session_label AS (
    SELECT
        user_session,
        MIN(CASE WHEN event_type = 'purchase' THEN event_ts ELSE NULL END) AS first_purchase_ts,
        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS target_purchase
    FROM sampled_events
    GROUP BY user_session
),

pre_purchase_events AS (
    SELECT
        e.*,
        l.first_purchase_ts,
        l.target_purchase,

        CASE
            WHEN e.event_type IN ('view', 'cart')
             AND (
                    l.first_purchase_ts IS NULL
                    OR e.event_ts < l.first_purchase_ts
                 )
            THEN 1
            ELSE 0
        END AS is_feature_event

    FROM sampled_events e
    JOIN session_label l
    ON e.user_session = l.user_session
),

agg AS (
    SELECT
        user_session,

        MIN(user_id) AS user_id,

        MIN(CASE WHEN is_feature_event = 1 THEN event_ts ELSE NULL END) AS session_start_ts,
        MAX(CASE WHEN is_feature_event = 1 THEN event_ts ELSE NULL END) AS session_end_ts,

        MIN(first_purchase_ts) AS first_purchase_ts,
        MAX(target_purchase) AS target_purchase,

        SUM(CASE WHEN is_feature_event = 1 AND event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
        SUM(CASE WHEN is_feature_event = 1 AND event_type = 'cart' THEN 1 ELSE 0 END) AS cart_count,

        COUNT(DISTINCT CASE WHEN is_feature_event = 1 THEN product_id ELSE NULL END) AS unique_product_count,
        COUNT(DISTINCT CASE WHEN is_feature_event = 1 THEN category_code ELSE NULL END) AS unique_category_count,
        COUNT(DISTINCT CASE WHEN is_feature_event = 1 THEN main_category ELSE NULL END) AS unique_main_category_count,
        COUNT(DISTINCT CASE WHEN is_feature_event = 1 THEN brand ELSE NULL END) AS unique_brand_count,

        AVG(CASE WHEN is_feature_event = 1 AND event_type = 'view' THEN price ELSE NULL END) AS avg_view_price,
        MAX(CASE WHEN is_feature_event = 1 AND event_type = 'view' THEN price ELSE NULL END) AS max_view_price,
        MIN(CASE WHEN is_feature_event = 1 AND event_type = 'view' THEN price ELSE NULL END) AS min_view_price,

        AVG(CASE WHEN is_feature_event = 1 AND event_type = 'cart' THEN price ELSE NULL END) AS avg_cart_price,
        MAX(CASE WHEN is_feature_event = 1 AND event_type = 'cart' THEN price ELSE NULL END) AS max_cart_price,
        MIN(CASE WHEN is_feature_event = 1 AND event_type = 'cart' THEN price ELSE NULL END) AS min_cart_price,
        SUM(CASE WHEN is_feature_event = 1 AND event_type = 'cart' THEN price ELSE 0 END) AS cart_total_price

    FROM pre_purchase_events
    GROUP BY user_session
),

final_features AS (
    SELECT
        user_session,
        user_id,

        session_start_ts,
        session_end_ts,
        first_purchase_ts,

        strftime(session_start_ts, '%Y_%m') AS session_month,

        COALESCE(view_count, 0) AS view_count,
        COALESCE(cart_count, 0) AS cart_count,

        CASE
            WHEN COALESCE(cart_count, 0) > 0 THEN 1
            ELSE 0
        END AS has_cart,

        COALESCE(view_count, 0) + COALESCE(cart_count, 0) AS pre_purchase_event_count,

        COALESCE(unique_product_count, 0) AS unique_product_count,
        COALESCE(unique_category_count, 0) AS unique_category_count,
        COALESCE(unique_main_category_count, 0) AS unique_main_category_count,
        COALESCE(unique_brand_count, 0) AS unique_brand_count,

        COALESCE(avg_view_price, 0) AS avg_view_price,
        COALESCE(max_view_price, 0) AS max_view_price,
        COALESCE(min_view_price, 0) AS min_view_price,
        COALESCE(max_view_price, 0) - COALESCE(min_view_price, 0) AS view_price_range,

        COALESCE(avg_cart_price, 0) AS avg_cart_price,
        COALESCE(max_cart_price, 0) AS max_cart_price,
        COALESCE(min_cart_price, 0) AS min_cart_price,
        COALESCE(max_cart_price, 0) - COALESCE(min_cart_price, 0) AS cart_price_range,
        COALESCE(cart_total_price, 0) AS cart_total_price,

        COALESCE(
            COALESCE(cart_count, 0) * 1.0 / NULLIF(COALESCE(view_count, 0), 0),
            0
        ) AS cart_view_ratio,

        COALESCE(
            (COALESCE(view_count, 0) + COALESCE(cart_count, 0)) * 1.0
            / NULLIF(COALESCE(unique_product_count, 0), 0),
            0
        ) AS events_per_product,

        COALESCE(
            COALESCE(avg_cart_price, 0) * 1.0 / NULLIF(COALESCE(avg_view_price, 0), 0),
            0
        ) AS cart_to_view_price_ratio,

        COALESCE(
            DATE_DIFF('second', session_start_ts, session_end_ts),
            0
        ) AS session_duration_seconds,

        COALESCE(EXTRACT(hour FROM session_start_ts), 0) AS start_hour,

        CASE
            WHEN strftime(session_start_ts, '%w') = '0' THEN 7
            ELSE COALESCE(CAST(strftime(session_start_ts, '%w') AS INTEGER), 0)
        END AS start_weekday_num,

        COALESCE(target_purchase, 0) AS target_purchase

    FROM agg
)

SELECT *
FROM final_features
WHERE
    session_start_ts IS NOT NULL
    AND pre_purchase_event_count > 0
    AND session_month IN (
        '2019_10', '2019_11', '2019_12',
        '2020_01', '2020_02', '2020_03', '2020_04'
    )
"""

print("\n开始构建防泄漏版 session 级行为特征，可能需要等待几分钟...")

con.execute(session_feature_sql)

print("sampled_session_features 创建完成。")

session_feature_check = con.execute("""
SELECT
    session_month,
    COUNT(*) AS session_count,
    SUM(target_purchase) AS purchase_session_count,
    AVG(target_purchase) AS purchase_session_ratio,
    AVG(view_count) AS avg_view_count,
    AVG(cart_count) AS avg_cart_count,
    AVG(pre_purchase_event_count) AS avg_pre_purchase_event_count,
    AVG(cart_view_ratio) AS avg_cart_view_ratio,
    AVG(session_duration_seconds) AS avg_session_duration_seconds
FROM sampled_session_features
GROUP BY session_month
ORDER BY session_month
""").df()

print("\n抽样后防泄漏 Session 特征检查：")
print(session_feature_check)

session_feature_check_csv = tables_dir / "model_v2_session_feature_check.csv"
session_feature_check_xlsx = tables_dir / "model_v2_session_feature_check.xlsx"

session_feature_check.to_csv(session_feature_check_csv, index=False, encoding="utf-8-sig")
session_feature_check.to_excel(session_feature_check_xlsx, index=False)

print("\n防泄漏 Session 特征检查表已保存：")
print(session_feature_check_csv)
print(session_feature_check_xlsx)

# =====================================================
# 10. 提取抽样用户列表
# =====================================================

print("\n开始提取抽样用户列表...")

con.execute("""
CREATE OR REPLACE TEMP TABLE sampled_users AS
SELECT DISTINCT
    user_id
FROM sampled_session_features
""")

sampled_user_count = con.execute("""
SELECT COUNT(*) AS sampled_user_count
FROM sampled_users
""").df()

print("\n抽样用户数量：")
print(sampled_user_count)

# =====================================================
# 11. 构建抽样用户的月度历史行为表
# =====================================================
# 只对抽样用户构建历史特征，避免对全部用户做历史聚合。
#
# 注意：
# 这里先构建 user_id + event_month 粒度的历史表。
# 后面给 session 匹配历史时，严格限制：
# h.event_month < s.session_month
#
# 即：
# 预测 2020_04 时，只能使用 2019_10 ~ 2020_03 的用户历史。
# 不能使用 2020_04 当月行为，也不能使用未来行为。

user_monthly_history_sql = """
CREATE OR REPLACE TEMP TABLE sampled_user_monthly_history AS
SELECT
    e.user_id,
    e.event_month,

    SUM(CASE WHEN e.event_type = 'view' THEN 1 ELSE 0 END) AS hist_month_view_count,
    SUM(CASE WHEN e.event_type = 'cart' THEN 1 ELSE 0 END) AS hist_month_cart_count,
    SUM(CASE WHEN e.event_type = 'purchase' THEN 1 ELSE 0 END) AS hist_month_purchase_count,

    SUM(CASE WHEN e.event_type = 'purchase' THEN e.price ELSE 0 END) AS hist_month_spend,

    COUNT(DISTINCT e.event_date) AS hist_month_active_day_count,

    COUNT(DISTINCT CASE WHEN e.event_type = 'view' THEN e.product_id ELSE NULL END) AS hist_month_view_product_count,
    COUNT(DISTINCT CASE WHEN e.event_type = 'cart' THEN e.product_id ELSE NULL END) AS hist_month_cart_product_count,
    COUNT(DISTINCT CASE WHEN e.event_type = 'purchase' THEN e.product_id ELSE NULL END) AS hist_month_purchase_product_count

FROM ecommerce_all e
JOIN sampled_users u
ON e.user_id = u.user_id

GROUP BY
    e.user_id,
    e.event_month
"""

print("\n开始构建抽样用户月度历史特征，可能需要等待几分钟...")

con.execute(user_monthly_history_sql)

print("sampled_user_monthly_history 创建完成。")

history_check = con.execute("""
SELECT
    event_month,
    COUNT(DISTINCT user_id) AS user_count,
    SUM(hist_month_view_count) AS view_count,
    SUM(hist_month_cart_count) AS cart_count,
    SUM(hist_month_purchase_count) AS purchase_count,
    SUM(hist_month_spend) AS spend
FROM sampled_user_monthly_history
GROUP BY event_month
ORDER BY event_month
""").df()

print("\n抽样用户历史行为检查：")
print(history_check)

history_check_csv = tables_dir / "model_v2_sampled_user_history_check.csv"
history_check_xlsx = tables_dir / "model_v2_sampled_user_history_check.xlsx"

history_check.to_csv(history_check_csv, index=False, encoding="utf-8-sig")
history_check.to_excel(history_check_xlsx, index=False)

print("\n抽样用户历史行为检查表已保存：")
print(history_check_csv)
print(history_check_xlsx)

# =====================================================
# 12. 构建每个 session 对应的历史特征
# =====================================================

user_history_before_session_sql = """
CREATE OR REPLACE TEMP TABLE user_history_before_session AS
SELECT
    s.user_session,

    COALESCE(SUM(h.hist_month_view_count), 0) AS user_hist_view_count,
    COALESCE(SUM(h.hist_month_cart_count), 0) AS user_hist_cart_count,
    COALESCE(SUM(h.hist_month_purchase_count), 0) AS user_hist_purchase_count,
    COALESCE(SUM(h.hist_month_spend), 0) AS user_hist_total_spend,

    COALESCE(SUM(h.hist_month_active_day_count), 0) AS user_hist_active_day_count,

    COALESCE(SUM(h.hist_month_view_product_count), 0) AS user_hist_view_product_count,
    COALESCE(SUM(h.hist_month_cart_product_count), 0) AS user_hist_cart_product_count,
    COALESCE(SUM(h.hist_month_purchase_product_count), 0) AS user_hist_purchase_product_count,

    COALESCE(COUNT(DISTINCT h.event_month), 0) AS user_hist_active_month_count,

    COALESCE(
        SUM(CASE WHEN h.hist_month_purchase_count > 0 THEN 1 ELSE 0 END),
        0
    ) AS user_hist_purchase_month_count

FROM sampled_session_features s

LEFT JOIN sampled_user_monthly_history h
ON s.user_id = h.user_id
AND h.event_month < s.session_month

GROUP BY
    s.user_session
"""

print("\n开始为每个 session 聚合历史特征...")

con.execute(user_history_before_session_sql)

print("user_history_before_session 创建完成。")

history_before_check = con.execute("""
SELECT
    COUNT(*) AS session_count,
    AVG(user_hist_view_count) AS avg_user_hist_view_count,
    AVG(user_hist_cart_count) AS avg_user_hist_cart_count,
    AVG(user_hist_purchase_count) AS avg_user_hist_purchase_count,
    AVG(user_hist_total_spend) AS avg_user_hist_total_spend,
    AVG(user_hist_active_month_count) AS avg_user_hist_active_month_count,
    AVG(user_hist_purchase_month_count) AS avg_user_hist_purchase_month_count
FROM user_history_before_session
""").df()

print("\nSession 历史特征整体检查：")
print(history_before_check)

# =====================================================
# 13. 合并 session 特征与用户历史特征
# =====================================================

model_dataset_sql = """
CREATE OR REPLACE TEMP TABLE model_dataset_all AS
SELECT
    CASE
        WHEN s.session_month IN ('2019_10', '2019_11', '2019_12', '2020_01', '2020_02') THEN 'train'
        WHEN s.session_month = '2020_03' THEN 'valid'
        WHEN s.session_month = '2020_04' THEN 'test'
        ELSE 'other'
    END AS dataset_type,

    s.user_session,
    s.user_id,
    s.session_month,

    -- session 行为特征
    COALESCE(s.view_count, 0) AS view_count,
    COALESCE(s.cart_count, 0) AS cart_count,
    COALESCE(s.has_cart, 0) AS has_cart,
    COALESCE(s.pre_purchase_event_count, 0) AS pre_purchase_event_count,

    COALESCE(s.unique_product_count, 0) AS unique_product_count,
    COALESCE(s.unique_category_count, 0) AS unique_category_count,
    COALESCE(s.unique_main_category_count, 0) AS unique_main_category_count,
    COALESCE(s.unique_brand_count, 0) AS unique_brand_count,

    -- session 价格特征
    COALESCE(s.avg_view_price, 0) AS avg_view_price,
    COALESCE(s.max_view_price, 0) AS max_view_price,
    COALESCE(s.min_view_price, 0) AS min_view_price,
    COALESCE(s.view_price_range, 0) AS view_price_range,

    COALESCE(s.avg_cart_price, 0) AS avg_cart_price,
    COALESCE(s.max_cart_price, 0) AS max_cart_price,
    COALESCE(s.min_cart_price, 0) AS min_cart_price,
    COALESCE(s.cart_price_range, 0) AS cart_price_range,
    COALESCE(s.cart_total_price, 0) AS cart_total_price,

    -- session 比例特征
    COALESCE(s.cart_view_ratio, 0) AS cart_view_ratio,
    COALESCE(s.events_per_product, 0) AS events_per_product,
    COALESCE(s.cart_to_view_price_ratio, 0) AS cart_to_view_price_ratio,

    -- 时间特征
    COALESCE(s.session_duration_seconds, 0) AS session_duration_seconds,
    COALESCE(s.start_hour, 0) AS start_hour,
    COALESCE(s.start_weekday_num, 0) AS start_weekday_num,

    -- 用户历史累计特征
    COALESCE(h.user_hist_view_count, 0) AS user_hist_view_count,
    COALESCE(h.user_hist_cart_count, 0) AS user_hist_cart_count,
    COALESCE(h.user_hist_purchase_count, 0) AS user_hist_purchase_count,
    COALESCE(h.user_hist_total_spend, 0) AS user_hist_total_spend,

    COALESCE(h.user_hist_active_day_count, 0) AS user_hist_active_day_count,
    COALESCE(h.user_hist_view_product_count, 0) AS user_hist_view_product_count,
    COALESCE(h.user_hist_cart_product_count, 0) AS user_hist_cart_product_count,
    COALESCE(h.user_hist_purchase_product_count, 0) AS user_hist_purchase_product_count,

    COALESCE(h.user_hist_active_month_count, 0) AS user_hist_active_month_count,
    COALESCE(h.user_hist_purchase_month_count, 0) AS user_hist_purchase_month_count,

    -- 用户历史比例特征
    COALESCE(
        COALESCE(h.user_hist_cart_count, 0) * 1.0 / NULLIF(COALESCE(h.user_hist_view_count, 0), 0),
        0
    ) AS user_hist_cart_view_ratio,

    COALESCE(
        COALESCE(h.user_hist_purchase_count, 0) * 1.0 / NULLIF(COALESCE(h.user_hist_cart_count, 0), 0),
        0
    ) AS user_hist_purchase_cart_ratio,

    COALESCE(
        COALESCE(h.user_hist_total_spend, 0) * 1.0 / NULLIF(COALESCE(h.user_hist_purchase_count, 0), 0),
        0
    ) AS user_hist_avg_spend_per_purchase,

    CASE
        WHEN COALESCE(h.user_hist_purchase_count, 0) > 0 THEN 1
        ELSE 0
    END AS user_is_old_customer,

    s.target_purchase

FROM sampled_session_features s
LEFT JOIN user_history_before_session h
ON s.user_session = h.user_session

WHERE s.session_month IN (
    '2019_10', '2019_11', '2019_12',
    '2020_01', '2020_02', '2020_03', '2020_04'
)
"""

print("\n开始合并 session 特征与用户历史特征...")

con.execute(model_dataset_sql)

print("model_dataset_all 创建完成。")

# =====================================================
# 14. 建模数据检查
# =====================================================

model_dataset_check_sql = """
SELECT
    dataset_type,
    session_month,
    COUNT(*) AS session_count,
    SUM(target_purchase) AS purchase_session_count,
    AVG(target_purchase) AS purchase_session_ratio,

    AVG(view_count) AS avg_view_count,
    AVG(cart_count) AS avg_cart_count,
    AVG(pre_purchase_event_count) AS avg_pre_purchase_event_count,
    AVG(cart_view_ratio) AS avg_cart_view_ratio,

    AVG(user_hist_purchase_count) AS avg_user_hist_purchase_count,
    AVG(user_hist_active_month_count) AS avg_user_hist_active_month_count,
    AVG(user_hist_purchase_month_count) AS avg_user_hist_purchase_month_count,
    AVG(user_is_old_customer) AS old_customer_ratio

FROM model_dataset_all
GROUP BY
    dataset_type,
    session_month
ORDER BY
    session_month
"""

model_dataset_check = con.execute(model_dataset_check_sql).df()

print("\n建模数据检查：")
print(model_dataset_check)

model_dataset_check_csv = tables_dir / "model_v2_dataset_check.csv"
model_dataset_check_xlsx = tables_dir / "model_v2_dataset_check.xlsx"

model_dataset_check.to_csv(model_dataset_check_csv, index=False, encoding="utf-8-sig")
model_dataset_check.to_excel(model_dataset_check_xlsx, index=False)

print("\n建模数据检查表已保存：")
print(model_dataset_check_csv)
print(model_dataset_check_xlsx)

# =====================================================
# 15. 输出特征列
# =====================================================

feature_columns = [
    # session 行为特征
    "view_count",
    "cart_count",
    "has_cart",
    "pre_purchase_event_count",

    "unique_product_count",
    "unique_category_count",
    "unique_main_category_count",
    "unique_brand_count",

    # session 价格特征
    "avg_view_price",
    "max_view_price",
    "min_view_price",
    "view_price_range",

    "avg_cart_price",
    "max_cart_price",
    "min_cart_price",
    "cart_price_range",
    "cart_total_price",

    # session 比例特征
    "cart_view_ratio",
    "events_per_product",
    "cart_to_view_price_ratio",

    # 时间特征
    "session_duration_seconds",
    "start_hour",
    "start_weekday_num",

    # 用户历史累计特征
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

    # 用户历史比例特征
    "user_hist_cart_view_ratio",
    "user_hist_purchase_cart_ratio",
    "user_hist_avg_spend_per_purchase",

    "user_is_old_customer"
]

feature_file = modeling_dir / "feature_columns.txt"

with open(feature_file, "w", encoding="utf-8") as f:
    for col in feature_columns:
        f.write(col + "\n")

print("\n特征列已保存：")
print(feature_file)

print("\n特征数量：")
print(len(feature_columns))

print("\n特征列：")
for col in feature_columns:
    print(col)

# =====================================================
# 16. 导出 train / valid / test 数据集
# =====================================================

train_output = modeling_dir / "train_dataset.parquet"
valid_output = modeling_dir / "valid_dataset.parquet"
test_output = modeling_dir / "test_dataset.parquet"

train_output_path = str(train_output).replace("\\", "/")
valid_output_path = str(valid_output).replace("\\", "/")
test_output_path = str(test_output).replace("\\", "/")

print("\n开始导出 train_dataset.parquet...")

con.execute(f"""
COPY (
    SELECT *
    FROM model_dataset_all
    WHERE dataset_type = 'train'
)
TO '{train_output_path}'
(
    FORMAT PARQUET,
    COMPRESSION 'ZSTD',
    OVERWRITE_OR_IGNORE TRUE
)
""")

print("train_dataset.parquet 导出完成。")

print("\n开始导出 valid_dataset.parquet...")

con.execute(f"""
COPY (
    SELECT *
    FROM model_dataset_all
    WHERE dataset_type = 'valid'
)
TO '{valid_output_path}'
(
    FORMAT PARQUET,
    COMPRESSION 'ZSTD',
    OVERWRITE_OR_IGNORE TRUE
)
""")

print("valid_dataset.parquet 导出完成。")

print("\n开始导出 test_dataset.parquet...")

con.execute(f"""
COPY (
    SELECT *
    FROM model_dataset_all
    WHERE dataset_type = 'test'
)
TO '{test_output_path}'
(
    FORMAT PARQUET,
    COMPRESSION 'ZSTD',
    OVERWRITE_OR_IGNORE TRUE
)
""")

print("test_dataset.parquet 导出完成。")

# =====================================================
# 17. 验证导出的数据集
# =====================================================

verify_sql = f"""
SELECT
    'train' AS dataset_type,
    COUNT(*) AS row_count,
    SUM(target_purchase) AS purchase_count,
    AVG(target_purchase) AS purchase_ratio
FROM read_parquet('{train_output_path}')

UNION ALL

SELECT
    'valid' AS dataset_type,
    COUNT(*) AS row_count,
    SUM(target_purchase) AS purchase_count,
    AVG(target_purchase) AS purchase_ratio
FROM read_parquet('{valid_output_path}')

UNION ALL

SELECT
    'test' AS dataset_type,
    COUNT(*) AS row_count,
    SUM(target_purchase) AS purchase_count,
    AVG(target_purchase) AS purchase_ratio
FROM read_parquet('{test_output_path}')
"""

verify_result = con.execute(verify_sql).df()

print("\n导出数据集验证：")
print(verify_result)

verify_csv = tables_dir / "model_v2_dataset_export_verify.csv"
verify_xlsx = tables_dir / "model_v2_dataset_export_verify.xlsx"

verify_result.to_csv(verify_csv, index=False, encoding="utf-8-sig")
verify_result.to_excel(verify_xlsx, index=False)

print("\n导出数据集验证表已保存：")
print(verify_csv)
print(verify_xlsx)

# =====================================================
# 18. 输出文件大小
# =====================================================

print("\n导出文件大小：")

for file in [train_output, valid_output, test_output]:
    size_mb = file.stat().st_size / 1024 / 1024
    print(f"{file.name}: {size_mb:.2f} MB")

# =====================================================
# 19. 输出最终说明
# =====================================================

print("\n第 14 步 V2：建模数据集构建完成。")

print("\n本版关键修改：")
print("1. 对购买 session，只使用首次 purchase 之前的 view/cart 构造特征，降低时间泄漏风险。")
print("2. 抽样比例提升为 train 2%、valid 3%、test 3%。")
print("3. 新增比例类特征、价格差异特征、用户历史活跃月份特征。")
print("4. 重新导出 train / valid / test 数据集，后续需要重新运行第 15 步和第 16 步。")