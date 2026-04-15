
# Проект: "Що впливає на самопочуття?"
# ============================================================
 
import pandas as pd
import numpy as np
 
# -- 1. ЗАВАНТАЖЕННЯ
 
df_screen       = pd.read_csv("data/src/ScreenTime vs MentalWellness.csv")
df_sleep        = pd.read_csv("data/src/Sleep_health_and_lifestyle_dataset.csv")
df_remote       = pd.read_csv("data/src/mental_health_remote_workers.csv")
df_productivity = pd.read_csv("data/src/mental_productivity_dataset.csv")
df_students     = pd.read_csv("data/src/student_lifestyle_dataset.csv")
df_survey       = pd.read_csv("data/src/survey.csv")
 
print("Файли завантажено")
print(f"  screen:       {df_screen.shape}")
print(f"  sleep:        {df_sleep.shape}")
print(f"  remote:       {df_remote.shape}")
print(f"  productivity: {df_productivity.shape}")
print(f"  students:     {df_students.shape}")
print(f"  survey:       {df_survey.shape}")
 
 
# -- 2. ОЧИЩЕННЯ
 
# screen: прибираємо порожній стовпець-артефакт, перейменовуємо
df_screen = df_screen.drop(columns=["Unnamed: 15"], errors="ignore")
df_screen = df_screen.rename(columns={
    "sleep_quality_1_5":          "sleep_quality",
    "stress_level_0_10":          "stress_level",
    "productivity_0_100":         "productivity",
    "exercise_minutes_per_week":  "exercise_min",
    "social_hours_per_week":      "social_hours",
    "mental_wellness_index_0_100":"wellness",
})
df_screen = df_screen.drop_duplicates()
 
# sleep: NaN у Sleep Disorder означає відсутність розладу
df_sleep["Sleep Disorder"] = df_sleep["Sleep Disorder"].fillna("None")
df_sleep.columns = df_sleep.columns.str.strip().str.lower().str.replace(" ", "_")
df_sleep = df_sleep.drop_duplicates()
 
# remote: прибираємо колонку з іменами
df_remote.columns = df_remote.columns.str.strip().str.lower()
df_remote = df_remote.drop(columns=["name"], errors="ignore")
df_remote = df_remote.drop_duplicates()
 
# productivity: скорочуємо довгі назви
df_productivity = df_productivity.rename(columns={
    "stress_level_1_10":       "stress_level",
    "productivity_score_1_10": "productivity",
    "mood_level_1_10":         "mood",
    "diet_quality_1_10":       "diet_quality",
})
df_productivity = df_productivity.drop(columns=["id"], errors="ignore")
df_productivity = df_productivity.drop_duplicates()
 
# students: кодуємо текстовий стрес у числовий для PCA
stress_map = {"Low": 1, "Moderate": 2, "High": 3}
df_students["stress_numeric"] = df_students["Stress_Level"].map(stress_map)
df_students = df_students.drop(columns=["Student_ID", "Stress_Level"], errors="ignore")
df_students = df_students.drop_duplicates()
 
# survey: залишаємо лише потрібні колонки, прибираємо рядки з багатьма пропусками
cols_keep = [
    "Age", "Gender", "treatment", "work_interfere",
    "remote_work", "benefits", "wellness_program",
    "mental_vs_physical", "family_history"
]
df_survey = df_survey[cols_keep].copy()
df_survey = df_survey[df_survey.isnull().sum(axis=1) <= 2]
for col in df_survey.select_dtypes(include="object").columns:
    df_survey[col] = df_survey[col].fillna(df_survey[col].mode()[0])
df_survey = df_survey[(df_survey["Age"] >= 15) & (df_survey["Age"] <= 80)]
df_survey = df_survey.drop_duplicates()
 
 
# -- 3. ДАТАСЕТ ДЛЯ PCA
 
pca_cols = [
    "sleep_hours",
    "sleep_quality",
    "stress_level",
    "screen_time_hours",
    "exercise_min",
    "social_hours",
    "productivity",
    "wellness",       # цільова змінна
]
 
df_pca = df_screen[pca_cols].dropna().copy()
 
print(f"\nДатасет для PCA: {df_pca.shape}")
print(df_pca.dtypes)
print("\nПерші 3 рядки:")
print(df_pca.head(3))
 
 
# -- 4. ЗБЕРЕЖЕННЯ
 
df_pca.to_csv("data/clean_data/clean_pca.csv", index=False)
df_sleep.to_csv("data/clean_data/clean_sleep.csv", index=False)
df_remote.to_csv("data/clean_data/clean_remote.csv", index=False)
df_productivity.to_csv("data/clean_data/clean_productivity.csv", index=False)
df_students.to_csv("data/clean_data/clean_students.csv", index=False)
df_survey.to_csv("data/clean_data/clean_survey.csv", index=False)
 
print("\nВсі очищені файли збережено у папку data/")