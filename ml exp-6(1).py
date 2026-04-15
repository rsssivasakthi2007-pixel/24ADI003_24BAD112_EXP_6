print("SIVASAKTHI S 24BAD112")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


data = pd.read_csv(r"C:\Users\priya\Downloads\diabetes_bagging.csv")

print("Dataset Preview:")
print(data.head())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:", dt_accuracy)


bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bagging_model.fit(X_train, y_train)

y_pred_bag = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, y_pred_bag)

print("Bagging Accuracy:", bagging_accuracy)


models = ["Decision Tree", "Bagging"]
accuracies = [dt_accuracy, bagging_accuracy]

plt.figure()

plt.bar(models, accuracies, label="Accuracy")

plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.title("Accuracy Comparison of Decision Tree vs Bagging")

plt.legend()


for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.show()

cm = confusion_matrix(y_test, y_pred_bag)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix - Bagging")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()
