"""
=============================================================
Healthcare Analytics Dashboard
=============================================================
Author:  Nasrum Bin Seron
Purpose: Exploratory analysis and visualisation of a simulated
         patient dataset covering admissions, diagnoses, length
         of stay, and readmission risk.

         Built as a teaching resource for the Healthcare Analytics
         module at Temasek Polytechnic, and as a portfolio piece
         demonstrating applied data analytics in a clinical context.

Libraries: pandas, matplotlib, seaborn, numpy
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── Output folder ─────────────────────────────────────────
OUTPUT_DIR = "output_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────
# 1. SIMULATE DATASET
#    In a real setting you would load a CSV here:
#    df = pd.read_csv("patient_data.csv")
# ─────────────────────────────────────────────────────────

N = 500  # number of patient records

diagnoses = [
    "Hypertension", "Diabetes Mellitus", "Acute Myocardial Infarction",
    "COPD", "Pneumonia", "Stroke", "Heart Failure", "Sepsis"
]

diagnosis_weights = [0.22, 0.20, 0.12, 0.10, 0.12, 0.08, 0.10, 0.06]

age_by_diagnosis = {
    "Hypertension": (65, 12),
    "Diabetes Mellitus": (58, 14),
    "Acute Myocardial Infarction": (62, 11),
    "COPD": (67, 10),
    "Pneumonia": (55, 18),
    "Stroke": (68, 13),
    "Heart Failure": (70, 11),
    "Sepsis": (60, 15),
}

los_by_diagnosis = {   # length of stay (days): (mean, std)
    "Hypertension": (3, 1.5),
    "Diabetes Mellitus": (4, 2.0),
    "Acute Myocardial Infarction": (7, 2.5),
    "COPD": (6, 2.0),
    "Pneumonia": (5, 2.5),
    "Stroke": (9, 3.0),
    "Heart Failure": (7, 2.5),
    "Sepsis": (10, 4.0),
}

diag_col      = rng.choice(diagnoses, size=N, p=diagnosis_weights)
age_col       = np.array([max(18, int(rng.normal(*age_by_diagnosis[d]))) for d in diag_col])
gender_col    = rng.choice(["Male", "Female"], size=N, p=[0.52, 0.48])
los_col       = np.array([max(1, int(rng.normal(*los_by_diagnosis[d]))) for d in diag_col])
ward_col      = rng.choice(["General", "ICU", "Step-down", "Isolation"], size=N,
                            p=[0.55, 0.15, 0.20, 0.10])

# Readmission: higher risk for older patients, longer stays, certain diagnoses
readmit_prob  = (
    0.05
    + 0.003 * (age_col - 50).clip(0)
    + 0.01  * (los_col - 4).clip(0)
    + np.where(np.isin(diag_col, ["Heart Failure", "COPD", "Sepsis"]), 0.15, 0)
).clip(0, 0.90)

readmit_col   = rng.binomial(1, readmit_prob).astype(bool)

df = pd.DataFrame({
    "patient_id":        [f"P{str(i).zfill(4)}" for i in range(1, N + 1)],
    "age":               age_col,
    "gender":            gender_col,
    "diagnosis":         diag_col,
    "length_of_stay":    los_col,
    "ward":              ward_col,
    "readmitted_30d":    readmit_col,
})

print("=" * 55)
print("HEALTHCARE ANALYTICS DASHBOARD")
print("=" * 55)
print(f"\nDataset: {N} patient records")
print(df.head())
print("\nData types:\n", df.dtypes)

# ─────────────────────────────────────────────────────────
# 2. SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────

print("\n── Summary Statistics ──────────────────────────────")
print(f"Mean age:                {df['age'].mean():.1f} years")
print(f"Mean length of stay:     {df['length_of_stay'].mean():.1f} days")
print(f"Overall readmission rate:{df['readmitted_30d'].mean() * 100:.1f}%")
print(f"\nDiagnosis breakdown:\n{df['diagnosis'].value_counts()}")
print(f"\nWard distribution:\n{df['ward'].value_counts()}")

readmit_by_diag = (
    df.groupby("diagnosis")["readmitted_30d"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "readmissions", "count": "total", "mean": "rate"})
    .sort_values("rate", ascending=False)
)
readmit_by_diag["rate_pct"] = (readmit_by_diag["rate"] * 100).round(1)
print(f"\nReadmission rate by diagnosis:\n{readmit_by_diag}")

# ─────────────────────────────────────────────────────────
# 3. VISUALISATIONS
# ─────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.05)

# ── Chart 1: Diagnosis Distribution ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
diag_counts = df["diagnosis"].value_counts()
bars = ax.barh(diag_counts.index, diag_counts.values,
               color=sns.color_palette("Blues_d", len(diag_counts)))
ax.set_xlabel("Number of Patients")
ax.set_title("Patient Volume by Primary Diagnosis", fontsize=14, fontweight="bold")
for bar, val in zip(bars, diag_counts.values):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=10)
ax.set_xlim(0, diag_counts.max() + 20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_diagnosis_distribution.png", dpi=150)
plt.close()
print("\nSaved: 01_diagnosis_distribution.png")

# ── Chart 2: Age Distribution by Gender ──────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for gender, color in zip(["Male", "Female"], ["#2171b5", "#fd8d3c"]):
    subset = df[df["gender"] == gender]["age"]
    ax.hist(subset, bins=20, alpha=0.6, label=gender, color=color, edgecolor="white")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Number of Patients")
ax.set_title("Age Distribution by Gender", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_age_distribution.png", dpi=150)
plt.close()
print("Saved: 02_age_distribution.png")

# ── Chart 3: Length of Stay by Diagnosis (Boxplot) ───────
fig, ax = plt.subplots(figsize=(11, 6))
order = df.groupby("diagnosis")["length_of_stay"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="diagnosis", y="length_of_stay", order=order,
            hue="diagnosis", palette="Blues", legend=False, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Length of Stay (days)")
ax.set_title("Length of Stay Distribution by Diagnosis", fontsize=14, fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_length_of_stay_boxplot.png", dpi=150)
plt.close()
print("Saved: 03_length_of_stay_boxplot.png")

# ── Chart 4: Readmission Rate by Diagnosis ────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#d73027" if r > 0.20 else "#2166ac"
          for r in readmit_by_diag["rate"]]
bars = ax.barh(readmit_by_diag.index, readmit_by_diag["rate_pct"], color=colors)
ax.axvline(df["readmitted_30d"].mean() * 100, color="black",
           linestyle="--", linewidth=1.2, label="Overall average")
ax.set_xlabel("30-day Readmission Rate (%)")
ax.set_title("30-day Readmission Rate by Diagnosis\n(red = above 20% threshold)",
             fontsize=14, fontweight="bold")
for bar, val in zip(bars, readmit_by_diag["rate_pct"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val}%", va="center", fontsize=10)
ax.set_xlim(0, readmit_by_diag["rate_pct"].max() + 8)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_readmission_rate.png", dpi=150)
plt.close()
print("Saved: 04_readmission_rate.png")

# ── Chart 5: Ward Utilisation ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ward_counts = df["ward"].value_counts()
wedges, texts, autotexts = ax.pie(
    ward_counts.values,
    labels=ward_counts.index,
    autopct="%1.1f%%",
    colors=sns.color_palette("Blues", len(ward_counts)),
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
for t in autotexts:
    t.set_fontsize(11)
ax.set_title("Ward Utilisation Breakdown", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_ward_utilisation.png", dpi=150)
plt.close()
print("Saved: 05_ward_utilisation.png")

# ── Chart 6: Age vs Length of Stay (scatter) ─────────────
fig, ax = plt.subplots(figsize=(9, 6))
colors_map = {"True": "#d73027", "False": "#2166ac"}
for readmit, group in df.groupby(df["readmitted_30d"].astype(str)):
    ax.scatter(group["age"], group["length_of_stay"],
               alpha=0.45, s=35,
               color=colors_map[readmit],
               label="Readmitted" if readmit == "True" else "Not readmitted")
m, b = np.polyfit(df["age"], df["length_of_stay"], 1)
x_line = np.linspace(df["age"].min(), df["age"].max(), 100)
ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5,
        linestyle="--", label=f"Trend (slope={m:.3f})")
ax.set_xlabel("Patient Age (years)")
ax.set_ylabel("Length of Stay (days)")
ax.set_title("Age vs Length of Stay\n(coloured by 30-day readmission)",
             fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_age_vs_los_scatter.png", dpi=150)
plt.close()
print("Saved: 06_age_vs_los_scatter.png")

# ─────────────────────────────────────────────────────────
# 4. EXPORT SUMMARY TABLE
# ─────────────────────────────────────────────────────────

summary = df.groupby("diagnosis").agg(
    total_patients=("patient_id", "count"),
    mean_age=("age", lambda x: round(x.mean(), 1)),
    mean_los=("length_of_stay", lambda x: round(x.mean(), 1)),
    readmission_rate_pct=("readmitted_30d", lambda x: round(x.mean() * 100, 1))
).sort_values("readmission_rate_pct", ascending=False)

summary.to_csv(f"{OUTPUT_DIR}/summary_by_diagnosis.csv")
print(f"\nSaved: summary_by_diagnosis.csv")

print("\n── All outputs saved to:", OUTPUT_DIR)
print("=" * 55)
print("Analysis complete.")
