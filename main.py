import pandas as pd
from analysis_utils import analyze_dataset

PLOTS_DIR = "data/plots"
CLEAN_DIR = "data/clean_data"

# 1. Main wellbeing dataset
analyze_dataset(
    pd.read_csv(f"{CLEAN_DIR}/clean_pca.csv"),
    target="wellness",
    dataset_name="PCA / Wellness",
    output_dir=PLOTS_DIR,
)

# 2. Productivity dataset
analyze_dataset(
    pd.read_csv(f"{CLEAN_DIR}/clean_productivity.csv"),
    target="productivity",
    dataset_name="Productivity",
    output_dir=PLOTS_DIR,
)

# 3. Sleep dataset -> target is stress
analyze_dataset(
    pd.read_csv(f"{CLEAN_DIR}/clean_sleep.csv"),
    target="stress_level",
    dataset_name="Sleep / Stress",
    output_dir=PLOTS_DIR,
)

# 4. Student dataset -> target is stress
analyze_dataset(
    pd.read_csv(f"{CLEAN_DIR}/clean_students.csv"),
    target="stress_numeric",
    dataset_name="Students / Stress",
    output_dir=PLOTS_DIR,
)

# 5. Remote work dataset -> target is productivity
analyze_dataset(
    pd.read_csv(f"{CLEAN_DIR}/clean_remote.csv"),
    target="productivity_score",
    dataset_name="Remote / Productivity",
    output_dir=PLOTS_DIR,
)
