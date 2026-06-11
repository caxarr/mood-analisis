# What Affects Well-Being? Data Analysis Project

**Published website:** https://caxarr.github.io/mood-analisis/

---

## Project Description

This project analyses which lifestyle, sleep, and work-related factors are
statistically associated with human well-being, stress, and productivity.
Five independently collected datasets are examined using Pearson correlation
and Principal Component Analysis (PCA).

Instead of merging all datasets into one, each dataset is analysed separately
to avoid misleading cross-population comparisons.

---

## Website Pages

| Page | File | Description |
|---|---|---|
| Home | `index.qmd` | Project overview and navigation |
| Analysis | `analysis.qmd` | Full statistical report (tables, figures, equations, citations) |
| Dashboard | `dashboard.qmd` | Interactive overview — key metrics at a glance |
| Presentation | `prezentation.qmd` | RevealJS slide deck |

---

## Datasets

All source files are in `data/src/`; cleaned versions used for analysis are
in `data/clean_data/`:

| File | Description | Target variable |
|---|---|---|
| `clean_pca.csv` | General lifestyle and wellness data | `wellness` |
| `clean_productivity.csv` | Productivity and daily habits (n = 20 000) | `productivity` |
| `clean_sleep.csv` | Clinical sleep and health study | `stress_level` |
| `clean_students.csv` | Student lifestyle and academic outcomes | `stress_numeric` |
| `clean_remote.csv` | Remote-work conditions and mental health | `productivity_score` |

Original sources (all from Kaggle):
- [Screen Time vs Mental Wellness Survey](https://www.kaggle.com/datasets/adharshinikumar/screentime-vs-mentalwellness-survey-2025)
- [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
- [Mental Health and Productivity Dataset](https://www.kaggle.com/datasets/dewminimnaadi/mental-health-and-productivity-dataset)
- [Student Lifestyle Dataset](https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset)
- [Mental Health in Tech Survey](https://www.kaggle.com/osmi/mental-health-in-tech-survey)
- [Mental Health of Remote Workers](https://www.kaggle.com/datasets/irhsa15/mental)

---

## Methods

- **Pearson correlation** — linear association between each predictor and the target
- **PCA** — dimensionality reduction to reveal dominant variance structure

---

## Project Structure

```
.
├── _quarto.yml             # Quarto website configuration
├── references.bib          # BibTeX citations
├── index.qmd               # Home page
├── analysis.qmd            # Main analysis report
├── dashboard.qmd           # Dashboard
├── prezentation.qmd        # RevealJS presentation
├── analysis_utils.py       # Python helper module (used by main.py)
├── cleaning.py             # Data cleaning scripts
├── main.py                 # Run full analysis pipeline
├── data/
│   ├── clean_data/         # Cleaned CSV datasets
│   ├── plots/              # Generated plots (from main.py)
│   └── src/                # Raw source CSVs
└── docs/                   # Rendered website output (GitHub Pages)
```

---

## How to Build

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter tabulate
```

Render the website:

```bash
quarto render
```

Output is placed in `docs/`. To preview locally:

```bash
quarto preview
```

---

## Author

Vladyslav Rudenko — Student project in data analysis.
