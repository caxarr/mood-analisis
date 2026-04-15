# What Affects Well-Being? Data Analysis Project

## Project description

This project focuses on analyzing factors that are related to human well-being, stress, mood, and productivity.
The main idea is to look at several different datasets and try to find common patterns between lifestyle variables such as sleep, physical activity, screen time, and mental state.

Instead of merging all datasets into one, each dataset is analyzed separately.
This allows avoiding incorrect assumptions, since the datasets come from different sources and represent different groups of people.

---

## Goal

The goal of this project is to understand:

- which variables are most strongly related to stress, productivity, and overall condition
- how different features behave together
- whether similar patterns appear across different datasets

The project does not aim to prove causation. It focuses on identifying relationships and structures in the data.

---

## Datasets

The analysis is based on the following cleaned datasets:

- `clean_pca.csv` — general lifestyle data (target: wellness)
- `clean_productivity.csv` — productivity and daily habits (target: productivity)
- `clean_sleep.csv` — sleep and health indicators (target: stress_level)
- `clean_students.csv` — student lifestyle data (target: stress_numeric)
- `clean_remote.csv` — remote work and mental state (target: productivity_score)

Each dataset is processed separately and uses its own target variable.

---

## Methods

### Data preparation

- removing duplicates
- handling missing values
- renaming columns
- selecting numeric features for analysis

---

### Correlation analysis

For each dataset, correlation between all variables and the selected target is calculated. This helps to identify which variables are most strongly associated with the target.

---

### Principal Component Analysis (PCA)

PCA is used to:

- reduce dimensionality
- identify main directions of variation in the data
- understand how variables group together

For each dataset, the following are analyzed:

- explained variance ratio
- loadings of the first principal component (PC1)
- relationship between the first two components

---

## Output

For each dataset, the following results are generated:

- correlation with target (bar chart)
- correlation matrix (heatmap)
- explained variance of principal components
- PC1 loadings
- PC1 vs PC2 scatter plot

All plots and tables are saved in:
- data/plots/

---

## How to run

1. Install dependencies:

- pip install pandas numpy matplotlib seaborn scikit-learn

2. Run the main script:

- python main.py

## Notes

- Datasets are not merged because they do not share a common structure or identifiers
- PCA is used to identify patterns, not to measure direct influence
- Results depend on the quality and structure of each dataset

---

## Conclusion

Across different datasets, some variables consistently appear to be related to human condition:

- stress level
- sleep duration and quality
- physical activity
- productivity or workload

These variables show strong correlations with target variables and also appear in the main principal components.

---

## Author

Vladyslav Rudenko
Student project in data analysis.
