from pathlib import Path
import duckdb
import shutil

# =====================================================
# 1. 定位项目路径
# =====================================================

current_file = Path(__file__).resolve()
project_dir = current_file.parents[1]

# 原始 CSV 仍然放在 data/raw
raw_dir = project_dir / "data" / "raw"

# 按你的要求，输出到 Data/Processed
# Windows 不区分大小写，所以 data/processed 和 Data/Processed 实际上指向同一类路径
processed_root = project_dir / "Data" / "Processed"

processed_root.mkdir(parents=True, exist_ok=True)

print("项目根目录：")
print(project_dir)

print("\n原始数据目录：")
print(raw_dir)

print("\nParquet 输出根目录：")
print(processed_root)

# =====================================================
# 2. 配置要处理的月份
# =====================================================

month_files = [
    {
        "csv_name": "2019-Oct.csv",
        "month_folder": "2019_10"
    },
    {
        "csv_name": "2019-Nov.csv",
        "month_folder": "2019_11"
    },
    {
        "csv_name": "2019-Dec.csv",
        "month_folder": "2019_12"
    },
    {
        "csv_name": "2020-Jan.csv",
        "month_folder": "2020_01"
    },
    {
        "csv_name": "2020-Feb.csv",
        "month_folder": "2020_02"
    },
    {
        "csv_name": "2020-Mar.csv",
        "month_folder": "2020_03"
    },
    {
        "csv_name": "2020-Apr.csv",
        "month_folder": "2020_04"
    }
]

# 如果对应月份文件夹已经存在，是否删除后重新导出
# True：重新生成
# False：检测到已存在就跳过该月份
OVERWRITE_EXISTING_MONTH = True

# 临时导出目录
temp_root = processed_root / "_tmp_partition_export"

# =====================================================
# 3. 检查原始文件是否存在
# =====================================================

print("\n开始检查原始 CSV 文件：")

valid_month_files = []

for item in month_files:
    raw_file = raw_dir / item["csv_name"]

    if raw_file.exists():
        size_gb = raw_file.stat().st_size / 1024 / 1024 / 1024
        print(f"{item['csv_name']} | {size_gb:.2f} GB | 存在")
        valid_month_files.append(item)
    else:
        print(f"{item['csv_name']} | 不存在，跳过")

if len(valid_month_files) == 0:
    raise FileNotFoundError("没有找到任何可处理的 CSV 文件，请检查 data/raw 目录。")

# =====================================================
# 4. 连接 DuckDB
# =====================================================

con = duckdb.connect()

# =====================================================
# 5. 逐月导出
# =====================================================

for item in valid_month_files:
    csv_name = item["csv_name"]
    month_folder = item["month_folder"]

    raw_file = raw_dir / csv_name
    month_output_dir = processed_root / month_folder
    temp_month_dir = temp_root / month_folder

    csv_path = str(raw_file).replace("\\", "/")
    temp_month_path = str(temp_month_dir).replace("\\", "/")

    print("\n" + "=" * 80)
    print(f"开始处理月份：{month_folder}")
    print(f"原始文件：{csv_path}")
    print(f"输出目录：{month_output_dir}")
    print("=" * 80)

    # -------------------------------------------------
    # 5.1 处理旧目录
    # -------------------------------------------------

    if month_output_dir.exists():
        if OVERWRITE_EXISTING_MONTH:
            print(f"\n检测到旧月份目录，准备删除：{month_output_dir}")
            shutil.rmtree(month_output_dir)
            print("旧月份目录已删除。")
        else:
            print(f"\n月份目录已存在，跳过该月份：{month_output_dir}")
            continue

    if temp_month_dir.exists():
        shutil.rmtree(temp_month_dir)

    temp_month_dir.mkdir(parents=True, exist_ok=True)
    month_output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 5.2 创建 DuckDB 视图
    # -------------------------------------------------
    # day_folder 格式：
    # 2019-10-01 -> 10.1
    # 2019-10-02 -> 10.2
    # 2020-04-30 -> 4.30

    con.execute(f"""
    CREATE OR REPLACE VIEW ecommerce_clean AS
    WITH base AS (
        SELECT
            TRY_CAST(REPLACE(CAST(event_time AS VARCHAR), ' UTC', '') AS TIMESTAMP) AS event_ts,
            event_type,
            product_id,
            category_id,
            COALESCE(category_code, 'unknown') AS category_code,
            COALESCE(brand, 'unknown') AS brand,
            CAST(price AS DOUBLE) AS price,
            user_id,
            COALESCE(user_session, 'unknown') AS user_session
        FROM read_csv_auto('{csv_path}', header=true)
        WHERE CAST(price AS DOUBLE) > 0
    )
    SELECT
        event_ts,
        CAST(event_ts AS DATE) AS event_date,
        '{month_folder}' AS event_month,
        (
            CAST(CAST(EXTRACT(month FROM event_ts) AS INTEGER) AS VARCHAR)
            || '.'
            || CAST(CAST(EXTRACT(day FROM event_ts) AS INTEGER) AS VARCHAR)
        ) AS day_folder,
        event_type,
        product_id,
        category_id,
        category_code,
        CASE
            WHEN category_code IS NULL OR category_code = 'unknown' THEN 'unknown'
            ELSE split_part(category_code, '.', 1)
        END AS main_category,
        brand,
        price,
        user_id,
        user_session,
        EXTRACT(hour FROM event_ts) AS hour,
        strftime(event_ts, '%A') AS weekday
    FROM base
    WHERE event_ts IS NOT NULL
    """)

    print("\nDuckDB 视图创建完成。")

    # -------------------------------------------------
    # 5.3 导出前检查
    # -------------------------------------------------

    check_sql = """
    SELECT
        MIN(event_ts) AS min_time,
        MAX(event_ts) AS max_time,
        COUNT(*) AS row_count,
        COUNT(DISTINCT event_date) AS date_count
    FROM ecommerce_clean
    """

    check_result = con.execute(check_sql).df()

    print("\n导出前数据检查：")
    print(check_result)

    daily_check_sql = """
    SELECT
        event_date,
        day_folder,
        COUNT(*) AS row_count
    FROM ecommerce_clean
    GROUP BY event_date, day_folder
    ORDER BY event_date
    """

    daily_check = con.execute(daily_check_sql).df()

    print("\n每日记录数检查：")
    print(daily_check)

    # -------------------------------------------------
    # 5.4 先导出到临时目录
    # -------------------------------------------------
    # DuckDB 的 PARTITION_BY 会生成 day_folder=10.1 这种目录
    # 后面我们再把它整理成 10.1

    print("\n开始导出到临时分区目录，可能需要等待几分钟...")

    con.execute(f"""
    COPY (
        SELECT
            event_ts,
            event_date,
            event_month,
            event_type,
            product_id,
            category_id,
            category_code,
            main_category,
            brand,
            price,
            user_id,
            user_session,
            hour,
            weekday,
            day_folder
        FROM ecommerce_clean
    )
    TO '{temp_month_path}'
    (
        FORMAT PARQUET,
        PARTITION_BY (day_folder),
        COMPRESSION 'ZSTD',
        OVERWRITE_OR_IGNORE TRUE
    )
    """)

    print("临时分区导出完成。")

    # -------------------------------------------------
    # 5.5 整理目录结构
    # -------------------------------------------------
    # 从：
    # _tmp_partition_export/2019_10/day_folder=10.1/*.parquet
    #
    # 改成：
    # Data/Processed/2019_10/10.1/part_001.parquet

    print("\n开始整理目录结构...")

    partition_dirs = sorted([
        p for p in temp_month_dir.iterdir()
        if p.is_dir() and p.name.startswith("day_folder=")
    ])

    if len(partition_dirs) == 0:
        raise RuntimeError(f"没有找到临时分区目录：{temp_month_dir}")

    for partition_dir in partition_dirs:
        day_folder = partition_dir.name.replace("day_folder=", "")
        target_day_dir = month_output_dir / day_folder
        target_day_dir.mkdir(parents=True, exist_ok=True)

        parquet_files = sorted(partition_dir.glob("*.parquet"))

        if len(parquet_files) == 0:
            print(f"警告：{partition_dir} 中没有 parquet 文件")
            continue

        for idx, parquet_file in enumerate(parquet_files, start=1):
            target_file = target_day_dir / f"part_{idx:03d}.parquet"
            shutil.move(str(parquet_file), str(target_file))

    print("目录结构整理完成。")

    # -------------------------------------------------
    # 5.6 删除临时目录
    # -------------------------------------------------

    if temp_month_dir.exists():
        shutil.rmtree(temp_month_dir)

    # -------------------------------------------------
    # 5.7 导出后检查
    # -------------------------------------------------

    output_parquet_files = list(month_output_dir.rglob("*.parquet"))
    output_size_mb = sum(file.stat().st_size for file in output_parquet_files) / 1024 / 1024

    print("\n导出后 Parquet 文件数量：")
    print(len(output_parquet_files))

    print("\n导出后 Parquet 总大小：")
    print(f"{output_size_mb:.2f} MB")

    print("\n前 10 个输出文件：")
    for file in output_parquet_files[:10]:
        print(file)

    # -------------------------------------------------
    # 5.8 验证导出后的 Parquet 可读
    # -------------------------------------------------

    parquet_glob_path = str(month_output_dir / "**" / "*.parquet").replace("\\", "/")

    verify_sql = f"""
    SELECT
        MIN(event_ts) AS min_time,
        MAX(event_ts) AS max_time,
        COUNT(*) AS row_count,
        COUNT(DISTINCT event_date) AS date_count
    FROM read_parquet('{parquet_glob_path}')
    """

    verify_result = con.execute(verify_sql).df()

    print("\nParquet 读取验证结果：")
    print(verify_result)

    # -------------------------------------------------
    # 5.9 示例：读取该月第一天数据
    # -------------------------------------------------

    first_day = daily_check.iloc[0]["day_folder"]
    first_day_dir = month_output_dir / first_day
    first_day_glob = str(first_day_dir / "*.parquet").replace("\\", "/")

    example_sql = f"""
    SELECT
        event_date,
        event_type,
        COUNT(*) AS event_count
    FROM read_parquet('{first_day_glob}')
    GROUP BY event_date, event_type
    ORDER BY event_type
    """

    example_result = con.execute(example_sql).df()

    print(f"\n示例：读取 {month_folder}/{first_day} 的数据并统计行为类型：")
    print(example_result)

    print(f"\n月份 {month_folder} 导出完成。")

# =====================================================
# 6. 清理总临时目录
# =====================================================

if temp_root.exists():
    shutil.rmtree(temp_root)

print("\n" + "=" * 80)
print("所有月份 CSV 转 Parquet 分区数据完成。")
print("=" * 80)

# =====================================================
# 7. 最终输出目录检查
# =====================================================

print("\n最终输出月份目录：")
for month_dir in sorted(processed_root.iterdir()):
    if month_dir.is_dir() and month_dir.name[:4].isdigit():
        day_dirs = [p for p in month_dir.iterdir() if p.is_dir()]
        parquet_count = len(list(month_dir.rglob("*.parquet")))
        size_mb = sum(file.stat().st_size for file in month_dir.rglob("*.parquet")) / 1024 / 1024

        print(
            f"{month_dir.name} | 日期文件夹数：{len(day_dirs)} | "
            f"Parquet 文件数：{parquet_count} | 大小：{size_mb:.2f} MB"
        )