# Healthcare Analytics Dashboard

A Python-based exploratory data analysis tool for patient admission data, built as a teaching resource for the **Healthcare Analytics** module at Temasek Polytechnic and as a portfolio demonstration of applied data analytics in a clinical context.

---

## What it does

Takes a patient dataset (simulated or real) covering admissions, diagnoses, length of stay, and readmission outcomes, then produces:

- Summary statistics and a readmission risk table
- 6 publication-ready charts
- A CSV summary report by diagnosis

---

## Charts produced

| Chart | Description |
|---|---|
| `01_diagnosis_distribution.png` | Patient volume by primary diagnosis |
| `02_age_distribution.png` | Age distribution split by gender |
| `03_length_of_stay_boxplot.png` | Length of stay by diagnosis (boxplot) |
| `04_readmission_rate.png` | 30-day readmission rate by diagnosis |
| `05_ward_utilisation.png` | Ward utilisation breakdown |
| `06_age_vs_los_scatter.png` | Age vs length of stay, coloured by readmission status |

---

## Sample output

Running the script on the default simulated dataset (500 patients, 8 diagnoses) produces findings including:

- Sepsis: 30.6% readmission rate (highest risk)
- COPD and Heart Failure: both above 20% threshold
- Mean length of stay: 5.3 days
- Clear positive correlation between age and length of stay

---

## Using your own data

Replace the simulation block with your actual dataset:

```python
# Replace this:
df = pd.DataFrame({ ... })   # simulated data

# With this:
df = pd.read_csv("your_patient_data.csv")
```

Your CSV should include columns: `age`, `gender`, `diagnosis`, `length_of_stay`, `ward`, `readmitted_30d`

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn
```

---

## Run

```bash
python healthcare_dashboard.py
```

Output charts saved to `output_charts/`

---

## Background

Built by **Nasrum Bin Seron**, Lecturer in Biomedical Engineering at Temasek Polytechnic. This project supports teaching in the Healthcare Analytics module and demonstrates the kind of data-driven decision support tools that are increasingly used in clinical operations, L&D programme evaluation, and healthcare workforce planning.

---

## Licence

MIT — free to use and adapt with attribution.
