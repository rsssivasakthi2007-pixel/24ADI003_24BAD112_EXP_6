print("SIVASAKTHI S 24BAD112")

# ✅ REMOVE WARNING
import os
import warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ---------------- LOAD DATA ----------------
df = pd.read_csv(r"C:\Users\priya\Downloads\fraud_smote.csv")

print("First 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

# ---------------- CLASS DISTRIBUTION BEFORE ----------------
print("\nClass Distribution BEFORE SMOTE:")
print(df['Fraud'].value_counts())

plt.figure()
df['Fraud'].value_counts().plot(kind='bar')
plt.title("Class Distribution BEFORE SMOTE")
plt.xlabel("Class (0=Normal, 1=Fraud)")
plt.ylabel("Count")
plt.show()

# ---------------- SPLIT DATA ----------------
X = df.drop('Fraud', axis=1)
y = df['Fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL BEFORE SMOTE ----------------
model_before = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

model_before.fit(X_train, y_train)

y_prob_before = model_before.predict_proba(X_test)[:, 1]

# 🔥 LOWER THRESHOLD
y_pred_before = (y_prob_before > 0.1).astype(int)

print("\nClassification Report BEFORE SMOTE:")
print(classification_report(y_test, y_pred_before, zero_division=0))

# ---------------- APPLY SMOTE ----------------
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nClass Distribution AFTER SMOTE:")
print(pd.Series(y_train_sm).value_counts())

plt.figure()
pd.Series(y_train_sm).value_counts().plot(kind='bar')
plt.title("Class Distribution AFTER SMOTE")
plt.xlabel("Class (0=Normal, 1=Fraud)")
plt.ylabel("Count")
plt.show()

# ---------------- MODEL AFTER SMOTE ----------------
model_after = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

model_after.fit(X_train_sm, y_train_sm)

y_prob_after = model_after.predict_proba(X_test)[:, 1]

# 🔥 LOWER THRESHOLD (IMPORTANT)
y_pred_after = (y_prob_after > 0.1).astype(int)

print("\nClassification Report AFTER SMOTE:")
print(classification_report(y_test, y_pred_after, zero_division=0))

# ---------------- PR CURVE ----------------
precision_before, recall_before, _ = precision_recall_curve(y_test, y_prob_before)
precision_after, recall_after, _ = precision_recall_curve(y_test, y_prob_after)

pr_auc_before = auc(recall_before, precision_before)
pr_auc_after = auc(recall_after, precision_after)

print("\nPR AUC Before SMOTE:", pr_auc_before)
print("PR AUC After SMOTE:", pr_auc_after)

# ---------------- PLOT PR CURVE ----------------
plt.figure()

plt.plot(recall_before, precision_before, linestyle='--',
         label=f"Before SMOTE (AUC={pr_auc_before:.4f})")

plt.plot(recall_after, precision_after, linestyle='-',
         label=f"After SMOTE (AUC={pr_auc_after:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Before vs After SMOTE)")
plt.legend()
plt.grid()
plt.show()
