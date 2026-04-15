print("SIVASAKTHI S 24BAD112")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\priya\Downloads\churn_boosting (1).csv")

data = data.dropna(subset=['Churn'])

for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].fillna('Unknown')

le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train, y_train)

gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)
y_pred_gb = gb.predict(X_test)

print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))

y_prob_ada = ada.predict_proba(X_test)[:, 1]
y_prob_gb = gb.predict_proba(X_test)[:, 1]

fpr_ada, tpr_ada, _ = roc_curve(y_test, y_prob_ada)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)

auc_ada = auc(fpr_ada, tpr_ada)
auc_gb = auc(fpr_gb, tpr_gb)

plt.figure()
plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost AUC = {auc_ada:.2f}')
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting AUC = {auc_gb:.2f}')
plt.plot([0,1],[0,1], linestyle='--', label='Random Model')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()


features = X.columns

ada_importances = ada.feature_importances_
ada_indices = np.argsort(ada_importances)

plt.figure()
plt.barh(features[ada_indices], ada_importances[ada_indices])
plt.title('Feature Importance (AdaBoost)')
plt.xlabel('Importance')
plt.show()

gb_importances = gb.feature_importances_
gb_indices = np.argsort(gb_importances)

plt.figure()
plt.barh(features[gb_indices], gb_importances[gb_indices])
plt.title('Feature Importance (Gradient Boosting)')
plt.xlabel('Importance')
plt.show()
print("\nClass Distribution:")
print(y.value_counts())
