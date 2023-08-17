import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from tabulate import tabulate
import numpy as np
from imblearn.over_sampling import SMOTENC

def missing_values(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    percentage_final = (round(percentage, 2) * 100)
    total_percent = pd.concat(objs=[total, percentage_final], axis=1, keys=['Total', '%'])
    return total_percent


df = pd.read_csv("Data/ILPD.csv")
df['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)


df['alkphos'].fillna(df['alkphos'].mean(), inplace=True)

dataset = df.to_numpy()

X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

fs = SelectKBest(score_func=f_classif, k='all')
fs.fit(X, y)

headers = ["age", "gender", "tot_bilirubin", "direct_bilirubin", "tot_proteins", "albumin", "ag_ratio", "sgpt", "sgot", "alkphos"]

scores = np.expand_dims(fs.scores_, axis=0)
scores_table = tabulate(scores, headers, floatfmt=".4f")

print("Anova F-score:\n", scores_table)

sm = SMOTENC(categorical_features=[0, 1], random_state=42)
X, y = sm.fit_resample(X, y)
fs.fit(X, y)
scores_after_resampling = np.expand_dims(fs.scores_, axis=0)
scores_after_resampling_table = tabulate(scores_after_resampling, headers, floatfmt=".4f")

print("\nAnova F-score after resampling:\n", scores_after_resampling_table)

fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X, y)
scores_chi2 = np.expand_dims(fs.scores_, axis=0)
scores_chi2 = tabulate(scores_chi2, headers, floatfmt=".4f")

print("\nChi2 F-score after resampling:\n", scores_chi2)