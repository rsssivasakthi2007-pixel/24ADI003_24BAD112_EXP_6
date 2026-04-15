In Scenario 1 (Bagging), a Decision Tree classifier is trained on the Diabetes dataset to predict whether a patient has diabetes. The performance of a single Decision Tree is compared with a BaggingClassifier to highlight how ensemble averaging improves model stability and accuracy. Visualizations such as accuracy comparison bar graphs and confusion matrices are used for evaluation.

In Scenario 2 (Boosting), customer churn prediction is performed using the Telco dataset. Two boosting techniques—AdaBoost and Gradient Boosting—are implemented and compared. The models are evaluated using ROC curves and feature importance plots to understand how boosting improves prediction by focusing on difficult samples.

In Scenario 3 (Random Forest), the Adult Income dataset is used to predict whether an individual earns more than 50K per year. A Random Forest model is trained, and the number of trees is tuned to observe its effect on performance. Feature importance and accuracy vs. number of trees graphs provide insights into model behavior and optimization.

In Scenario 4 (Stacking), heart disease prediction is carried out using multiple base models including Logistic Regression, Support Vector Machine, and Decision Tree. These models are combined using a StackingClassifier to demonstrate how meta-learning can improve predictive performance. A comparison bar chart shows the effectiveness of stacking over individual models.

In Scenario 5 (SMOTE), the Credit Card Fraud Detection dataset is used to address class imbalance issues. The dataset is analyzed for imbalance, and SMOTE (Synthetic Minority Oversampling Technique) is applied to generate synthetic samples. Models are trained before and after SMOTE to compare performance improvements. Visualizations include class distribution plots and precision-recall curves.

# 24ADI003_24BAD112_EXP_6
