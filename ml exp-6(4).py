print("SIVASAKTHI S 24BAD112")
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

df = pd.read_csv(r"C:\Users\priya\Downloads\heart_stacking.csv") 

print("First 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

X = df[['Cholesterol', 'MaxHeartRate', 'Age']]
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
svm = SVC(probability=True)
dt = DecisionTreeClassifier()

lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
dt.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
svm_pred = svm.predict(X_test)
dt_pred = dt.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
svm_acc = accuracy_score(y_test, svm_pred)
dt_acc = accuracy_score(y_test, dt_pred)

estimators = [
    ('lr', LogisticRegression()),
    ('svm', SVC(probability=True)),
    ('dt', DecisionTreeClassifier())
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stack.fit(X_train, y_train)
stack_pred = stack.predict(X_test)

stack_acc = accuracy_score(y_test, stack_pred)

print("\nModel Accuracies:")
print(f"Logistic Regression: {lr_acc:.4f}")
print(f"SVM: {svm_acc:.4f}")
print(f"Decision Tree: {dt_acc:.4f}")
print(f"Stacking: {stack_acc:.4f}")

models = ['Logistic Regression', 'SVM', 'Decision Tree', 'Stacking']
accuracies = [lr_acc, svm_acc, dt_acc, stack_acc]

plt.figure()
plt.bar(models, accuracies)

plt.title("Model Comparison (Heart Disease Prediction)")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
