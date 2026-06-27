# GitHub Upload Notes

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
