from pathlib import Path
import shutil

# =====================================================
# 1. 定位项目路径
# =====================================================

current_file = Path(__file__).resolve()
project_dir = current_file.parents[1]

release_dir = project_dir.parent / "Ecommerce_behavior_project_github"

CLEAR_EXISTING_RELEASE_DIR = True

print("原项目目录：")
print(project_dir)

print("\nGitHub 发布目录：")
print(release_dir)

# =====================================================
# 2. 清理旧发布目录
# =====================================================

if release_dir.exists() and CLEAR_EXISTING_RELEASE_DIR:
    print("\n检测到旧 GitHub 发布目录，正在删除：")
    print(release_dir)
    shutil.rmtree(release_dir)

release_dir.mkdir(parents=True, exist_ok=True)

# =====================================================
# 3. 工具函数
# =====================================================

def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"[跳过] 文件不存在：{src}")
        return False

    if not src.is_file():
        print(f"[跳过] 不是文件：{src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    size_kb = dst.stat().st_size / 1024
    print(f"[复制] {src.relative_to(project_dir)} -> {dst.relative_to(release_dir)} | {size_kb:.2f} KB")
    return True


def copy_relative_file(relative_path: str) -> bool:
    src = project_dir / relative_path
    dst = release_dir / relative_path
    return copy_file(src, dst)


def write_text_file(relative_path: str, content: str):
    dst = release_dir / relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    print(f"[写入] {dst.relative_to(release_dir)}")


# =====================================================
# 4. 复制根目录文档
# =====================================================

root_files = [
    "README.md",
    "PROJECT_SUMMARY.md",
    "RESUME_DESCRIPTION.md",
    "requirements.txt",
]

print("\n========== 复制根目录文档 ==========")

for file in root_files:
    copy_relative_file(file)

# =====================================================
# 5. 复制正式主流程脚本
# =====================================================
# 只复制最终项目主线，不复制 01~10 早期探索脚本，也不复制旧版 11_Export_Full_Month_Parquet.py

core_scripts = [
    "Notebook/11_Export_All_Months_Parquet.py",
    "Notebook/12_Multi_Month_KPI_Trend.py",
    "Notebook/13_Cohort_Retention_Analysis.py",
    "Notebook/14_Build_Model_Dataset_DuckDB.py",
    "Notebook/15_Train_LightGBM_Time_Split.py",
    "Notebook/16_Model_Evaluation_And_Interpretation.py",
    "Notebook/17_Generate_Project_Summary.py",
    "Notebook/18_Prepare_GitHub_Release.py",
]

print("\n========== 复制正式主流程脚本 ==========")

for script in core_scripts:
    copy_relative_file(script)

# =====================================================
# 6. 复制关键图表
# =====================================================

figure_files = [
    # Cohort / 复购分析
    "outputs/figures/cohort_retention_heatmap.png",
    "outputs/figures/monthly_new_repeat_purchase_users.png",
    "outputs/figures/monthly_repeat_user_ratio.png",

    # 最终 LightGBM 模型
    "outputs/figures/lightgbm_test_roc_curve.png",
    "outputs/figures/lightgbm_test_pr_curve.png",
    "outputs/figures/lightgbm_test_lift.png",
    "outputs/figures/lightgbm_feature_importance_top20.png",

    # V2 模型对比
    "outputs/figures/model_v2_comparison_test_auc.png",
    "outputs/figures/model_v2_comparison_test_pr_auc.png",
    "outputs/figures/model_v2_comparison_top10_lift.png",
]

print("\n========== 复制关键图表 ==========")

for fig in figure_files:
    copy_relative_file(fig)

# =====================================================
# 7. 复制关键结果表
# =====================================================

table_files = [
    # 多月 KPI
    "outputs/tables/monthly_kpi_summary.xlsx",
    "outputs/tables/monthly_kpi_summary.csv",

    # Cohort 留存与复购
    "outputs/tables/cohort_summary.xlsx",
    "outputs/tables/cohort_summary.csv",
    "outputs/tables/cohort_retention_rate_matrix.xlsx",
    "outputs/tables/cohort_retention_rate_matrix.csv",
    "outputs/tables/monthly_new_repeat_purchase_users.xlsx",
    "outputs/tables/monthly_new_repeat_purchase_users.csv",

    # V2 建模数据检查
    "outputs/tables/model_v2_dataset_check.xlsx",
    "outputs/tables/model_v2_dataset_check.csv",
    "outputs/tables/model_v2_dataset_export_verify.xlsx",
    "outputs/tables/model_v2_dataset_export_verify.csv",

    # 最终 LightGBM 结果
    "outputs/tables/lightgbm_time_split_model_metrics.xlsx",
    "outputs/tables/lightgbm_time_split_model_metrics.csv",
    "outputs/tables/lightgbm_lift_analysis.xlsx",
    "outputs/tables/lightgbm_lift_analysis.csv",
    "outputs/tables/lightgbm_feature_importance.xlsx",
    "outputs/tables/lightgbm_feature_importance.csv",
    "outputs/tables/lightgbm_test_confusion_matrix.xlsx",
    "outputs/tables/lightgbm_test_confusion_matrix.csv",

    # V2 模型对比、解释、错误分析
    "outputs/tables/model_v2_ablation_summary.xlsx",
    "outputs/tables/model_v2_ablation_summary.csv",
    "outputs/tables/model_v2_feature_group_importance.xlsx",
    "outputs/tables/model_v2_feature_group_importance.csv",
    "outputs/tables/model_v2_error_analysis_by_group.xlsx",
    "outputs/tables/model_v2_error_analysis_by_group.csv",
    "outputs/tables/model_v2_high_score_session_profile.xlsx",
    "outputs/tables/model_v2_high_score_session_profile.csv",

    # 汇总解释
    "outputs/tables/model_interpretation_summary.xlsx",
    "outputs/tables/model_interpretation_summary.csv",
    "outputs/tables/feature_business_interpretation.xlsx",
    "outputs/tables/feature_business_interpretation.csv",
    "outputs/tables/model_business_recommendations.txt",
]

print("\n========== 复制关键结果表 ==========")

for table in table_files:
    copy_relative_file(table)

# =====================================================
# 8. 写入 data / Data / models 占位说明
# =====================================================
# 注意：
# 这里使用 ~~~text / ~~~bash，而不是 ```text / ```bash。
# 这样可以避免在 ChatGPT 页面中出现 Markdown 代码块嵌套显示错误。

data_readme = """# Data Directory

This directory is intentionally kept empty in the GitHub version.

The raw CSV files, generated Parquet partitions, and modeling Parquet datasets are too large to upload to GitHub.

Expected local structure:

~~~text
data/
├── raw/
├── processed/
└── modeling/
    ├── train_dataset.parquet
    ├── valid_dataset.parquet
    ├── test_dataset.parquet
    └── feature_columns.txt
~~~

To reproduce the project locally, download the raw dataset and run the scripts in `Notebook/` in order.
"""

processed_data_readme = """# Processed Data Directory

The processed Parquet partitions are intentionally excluded from GitHub.

Expected local structure:

~~~text
Data/Processed/
├── 2019_10/
├── 2019_11/
├── 2019_12/
├── 2020_01/
├── 2020_02/
├── 2020_03/
└── 2020_04/
~~~

These files can be regenerated by running:

~~~bash
python Notebook/11_Export_All_Months_Parquet.py
~~~
"""

models_readme = """# Models Directory

Model files are intentionally excluded from GitHub.

The final LightGBM model can be regenerated by running:

~~~bash
python Notebook/15_Train_LightGBM_Time_Split.py
~~~

Excluded model files include:

~~~text
*.pkl
*.joblib
~~~
"""

print("\n========== 写入占位说明文件 ==========")

write_text_file("data/README.md", data_readme)
write_text_file("Data/README.md", processed_data_readme)
write_text_file("models/README.md", models_readme)

# =====================================================
# 9. 写入 GitHub 专用 .gitignore
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
Data/Processed/
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

print("\n========== 写入 .gitignore ==========")

write_text_file(".gitignore", gitignore_text)

# =====================================================
# 10. 写入 GitHub 上传说明
# =====================================================

upload_notes = """# GitHub Upload Notes

This is a cleaned GitHub release version of the project.

## Included

- README.md
- PROJECT_SUMMARY.md
- RESUME_DESCRIPTION.md
- requirements.txt
- Notebook core scripts
- selected figures
- selected summary tables
- data / Data / models placeholder README files

## Excluded

Large or reproducible files are excluded:

- raw CSV files
- Parquet files
- DuckDB temporary files
- model pkl / joblib files
- large prediction detail files

## Main Workflow

Run the scripts in the following order:

~~~bash
python Notebook/11_Export_All_Months_Parquet.py
python Notebook/12_Multi_Month_KPI_Trend.py
python Notebook/13_Cohort_Retention_Analysis.py
python Notebook/14_Build_Model_Dataset_DuckDB.py
python Notebook/15_Train_LightGBM_Time_Split.py
python Notebook/16_Model_Evaluation_And_Interpretation.py
python Notebook/17_Generate_Project_Summary.py
~~~

## Final Model Result

The final LightGBM full model achieves:

- ROC-AUC: 0.9718
- PR-AUC: 0.6342
- F1-score: 0.6424
- Top 10% purchase rate: 54.03%
- Top 10% Lift: 7.97
- Captured purchase sessions in Top 10%: 79.72%
"""

print("\n========== 写入 GitHub 上传说明 ==========")

write_text_file("GITHUB_UPLOAD_NOTES.md", upload_notes)

# =====================================================
# 11. 检查是否混入禁止上传的大文件类型
# =====================================================

forbidden_suffixes = {
    ".parquet",
    ".pkl",
    ".joblib",
    ".duckdb",
    ".db",
}

forbidden_keywords = [
    "data/raw",
    "Data/Processed",
    "duckdb_temp",
]

forbidden_files = []

for file in release_dir.rglob("*"):
    if not file.is_file():
        continue

    relative = file.relative_to(release_dir).as_posix()

    if file.suffix.lower() in forbidden_suffixes:
        forbidden_files.append(relative)
        continue

    for keyword in forbidden_keywords:
        if keyword in relative:
            forbidden_files.append(relative)
            break

# =====================================================
# 12. 输出目录统计
# =====================================================

file_count = 0
total_size = 0

for file in release_dir.rglob("*"):
    if file.is_file():
        file_count += 1
        total_size += file.stat().st_size

total_size_mb = total_size / 1024 / 1024

print("\n========== GitHub 发布目录整理完成 ==========")

print(f"发布目录：{release_dir}")
print(f"文件数量：{file_count}")
print(f"总大小：{total_size_mb:.2f} MB")

print("\n========== 禁止文件检查 ==========")

if len(forbidden_files) == 0:
    print("未发现 parquet / pkl / joblib / duckdb / 原始数据目录等禁止上传内容。")
else:
    print("警告：发现疑似不应上传的文件：")
    for file in forbidden_files:
        print(file)

print("\n========== 发布目录结构预览 ==========")

for path in sorted(release_dir.rglob("*")):
    relative = path.relative_to(release_dir)

    # 只展示前 3 层，避免输出过长
    depth = len(relative.parts)

    if depth <= 3:
        if path.is_dir():
            print(f"[DIR ] {relative}")
        else:
            size_kb = path.stat().st_size / 1024
            print(f"[FILE] {relative} | {size_kb:.2f} KB")

print("\n下一步：")
print("1. 打开 PowerShell")
print(f"2. cd /d {release_dir}")
print("3. 运行 git init / git add . / git commit")