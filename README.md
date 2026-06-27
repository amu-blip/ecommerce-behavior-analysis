# 多月电商用户行为分析与购买转化预测
## Project Highlights

* Processed **410M+** e-commerce user behavior events across **7 months** using DuckDB and Parquet.
* Built a scalable data pipeline from raw CSV files to monthly and daily Parquet partitions.
* Designed leakage-safe session-level features by using only pre-purchase `view/cart` behavior and historical user behavior before the target month.
* Trained and compared has-cart rule baseline, Logistic Regression, and multiple LightGBM ablation models.
* Final LightGBM full model achieved **ROC-AUC 0.9718**, **PR-AUC 0.6342**, and **F1-score 0.6424** on the 2020-04 test set.
* Top 10% high-score sessions reached **54.03% purchase rate**, **7.97x Lift**, and captured **79.72%** of all real purchase sessions.

## 项目亮点

* 使用 DuckDB 和 Parquet 处理 **7 个月、4.11 亿+** 条电商用户行为日志。
* 构建从原始 CSV 到按月份 / 日期分区 Parquet 的大规模数据处理流程。
* 设计防泄漏 session 级特征：购买 session 仅使用首次购买前的 `view/cart` 行为，用户历史特征仅使用目标月份之前的数据。
* 对比 has-cart 规则 baseline、Logistic Regression 和多组 LightGBM ablation 模型。
* 最终 LightGBM full model 在 2020-04 测试集取得 **ROC-AUC 0.9718**、**PR-AUC 0.6342**、**F1-score 0.6424**。
* 模型预测概率最高的前 10% session 购买率达到 **54.03%**，Lift 为 **7.97 倍**，覆盖 **79.72%** 的真实购买 session。

## 1. 项目概述

本项目基于多品类电商平台用户行为日志，围绕 `view`、`cart`、`purchase` 三类核心行为，完成从大规模数据处理、经营指标分析、用户留存分析到 session 级购买预测建模的完整流程。

项目重点不是单纯做 EDA，而是构建一条可复用的数据分析与建模链路：

```text
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
```

## 2. 数据规模

| 指标 | 数值 |
| --- | --- |
| 时间范围 | 2019-10 至 2020-04 |
| 月份数 | 7 |
| 天数 | 213 |
| 有效行为记录数 | 410,995,046 |
| 用户数 | 15,635,832 |
| Session 数 | 89,614,715 |
| 购买事件数 | 6,848,824 |
| 购买用户数 | 2,064,899 |

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

```text
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
```

## 5. 数据工程

原始数据为多个大体量 CSV 文件，不适合在每个分析脚本中反复使用 pandas 直接读取。因此，本项目使用 DuckDB 对原始行为日志进行清洗，并导出为 Parquet 分区数据。

Parquet 目录按月份和日期组织：

```text
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
```

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

- 销售额最高月份为 **2020_02**，销售额约 **381,178,935.42**。
- 购买次数最高月份购买事件数约 **1,200,288**。
- 最高购买 / 浏览比例约为 **2.35%**。

这些指标用于识别不同月份的交易峰值、转化效率变化和用户购买行为变化。

## 7. Cohort 留存与复购分析

项目基于用户首次购买月份构建 cohort 留存矩阵，分析不同首购月份用户在后续月份的复购表现。

核心发现：

- 次月留存率最高的 cohort 为 **2019_10**，次月留存率达到 **26.30%**。
- 首购用户规模最大的 cohort 为 **2019_11**，首购用户数为 **350,352**。
- 复购用户占比最高月份为 **2020_03**，复购用户占比达到 **42.96%**。

该部分说明，用户规模和用户质量并不完全一致，首购用户数量高并不必然意味着后续留存更好。

## 8. Session 级购买预测建模

### 8.1 建模目标

将用户行为日志聚合为 session 级样本，预测一个 session 是否会发生购买：

```text
target_purchase = 该 session 是否出现 purchase 行为
```

这是一个典型的类别不平衡二分类问题。测试集中购买 session 占比约为 **6.78%**。

### 8.2 时间切分

为模拟真实业务中“用历史数据预测未来月份”的场景，项目采用时间切分，而不是随机切分。

| 数据集 | 时间范围 | 样本数 | 购买样本占比 |
| --- | --- | --- | --- |
| Train | 2019-10 至 2020-02 | 1,310,243 | 5.76% |
| Valid | 2020-03 | 378,088 | 6.67% |
| Test | 2020-04 | 348,290 | 6.78% |

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

```text
view -> cart -> purchase -> view -> cart
```

在构造特征时，只使用：

```text
view -> cart
```

购买后的行为不会进入模型特征。

## 9. 模型训练与对比

项目训练并比较了多个模型：

| 模型 | 说明 | Top 10% Lift |
| --- | --- | --- |
| has_cart rule | 只根据是否加购判断购买倾向 | 6.36 |
| Logistic Regression | 线性模型，使用全部特征 | 7.07 |
| LightGBM behavior only | 只使用当前 session 行为特征 | 7.92 |
| LightGBM behavior + price + time | 加入价格和时间特征 | 7.89 |
| LightGBM full | 加入用户历史特征，最终模型 | 7.97 |

最终选择 **LightGBM full model**，因为它在测试集上的 ROC-AUC、PR-AUC、F1 和 Top 10% Lift 综合表现最好。

## 10. 最终模型表现

最终模型在 2020-04 测试集上的表现如下：

| 指标 | 测试集结果 |
| --- | --- |
| ROC-AUC | 0.9718 |
| PR-AUC | 0.6342 |
| Best F1 Threshold | 0.49 |
| Accuracy | 0.9428 |
| Precision | 0.5573 |
| Recall | 0.7582 |
| F1-score | 0.6424 |

由于购买样本占比较低，项目不仅关注 accuracy，也重点关注 PR-AUC、Recall、F1 和 Lift。

## 11. Lift 分析

在电商营销场景中，模型的排序能力通常比单纯二分类更重要。本项目使用 Lift 分析衡量模型识别高购买概率 session 的能力。

| 分组 | 购买率 | Lift | 捕获真实购买比例 |
| --- | --- | --- | --- |
| 整体测试集 | 6.78% | 1.00 | 100.00% |
| Top 1% 高分 session | 77.46% | NA | NA |
| Top 5% 高分 session | 66.29% | NA | NA |
| Top 10% 高分 session | 54.03% | 7.97 | 79.72% |

核心结论：

- 测试集整体购买率为 **6.78%**。
- 模型预测概率最高的前 10% session 中，真实购买率达到 **54.03%**。
- Top 10% Lift 为 **7.97 倍**。
- Top 10% 高分 session 覆盖了 **79.72%** 的真实购买 session。

这说明模型适合用于营销资源排序，例如优惠券、客服触达、推荐位分配和购物车召回优先级排序。

## 12. 特征重要性

### 12.1 特征组重要性

| 特征组 | 特征数 | 重要性占比 |
| --- | --- | --- |
| 当前 session 加购行为 | 3 | 86.16% |
| 用户历史活跃特征 | 7 | 5.68% |
| 当前 session 价格特征 | 10 | 2.66% |
| 用户历史购买特征 | 7 | 2.11% |
| 当前 session 浏览/探索行为 | 7 | 1.80% |
| 当前 session 时间特征 | 3 | 1.59% |

当前 session 加购行为是最重要的特征组，重要性占比约 **86.16%**。这说明购买预测主要由加购行为驱动，同时用户历史活跃度、价格特征和历史购买行为也提供了增量信息。

### 12.2 Top 特征

| 特征 | 重要性占比 |
| --- | --- |
| cart_count | 78.96% |
| has_cart | 7.00% |
| user_hist_active_month_count | 2.90% |
| user_hist_active_day_count | 2.41% |
| avg_cart_price | 1.66% |
| session_duration_seconds | 1.29% |
| events_per_product | 1.17% |
| user_hist_purchase_count | 0.82% |
| user_hist_total_spend | 0.72% |
| user_hist_purchase_cart_ratio | 0.53% |

最重要的单个特征是 **cart_count**，说明首次购买前的加购次数是购买意向最核心的行为信号。

## 13. 错误分析

使用验证集选择的最佳 F1 阈值 **0.49** 后，测试集错误类型如下：

| 错误类型 | 样本数 | 含义 |
| --- | --- | --- |
| TN | 310,466 | 实际未购买，模型也判断为未购买 |
| TP | 17,899 | 实际购买，模型也判断为购买 |
| FP | 14,218 | 模型判断会购买，但实际未购买 |
| FN | 5,707 | 模型判断不会购买，但实际购买 |

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

```bash
python Notebook/11_Export_All_Months_Parquet.py
python Notebook/12_Multi_Month_KPI_Trend.py
python Notebook/13_Cohort_Retention_Analysis.py
python Notebook/14_Build_Model_Dataset_DuckDB.py
python Notebook/15_Train_LightGBM_Time_Split.py
python Notebook/16_Model_Evaluation_And_Interpretation.py
python Notebook/17_Generate_Project_Summary.py
```

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
