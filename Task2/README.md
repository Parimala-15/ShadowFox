# 🏦 Loan Price Prediction using SVM

This project predicts loan approval status based on various applicant features using a **Support Vector Machine (SVM)** classifier. The dataset contains missing values, which are handled using **mean imputation** with `SimpleImputer`.

---

## 📌 Project Overview

- **Goal**: Predict whether a loan will be approved or not.
- **Model Used**: Support Vector Classifier (`SVC`) from Scikit-learn.
- **Data Cleaning**: Missing values handled using `SimpleImputer`.
- **Evaluation Metric**: Accuracy Score on Training and Test datasets.

---

## 📁 Files Included

- `loan_prediction.ipynb`: Jupyter Notebook containing all code.
- `README.md`: Project documentation (you are here).
- `dataset.csv`: (Add if available)

---

## ⚙️ Libraries Used
numpy
pandas
scikit-learn (sklearn)


---

## 🧠 ML Workflow

1. **Import libraries and dataset**
2. **Handle missing values** using `SimpleImputer(strategy='mean')`
3. **Split the dataset** into training and testing sets
4. **Train the model** using `SVC()`
5. **Evaluate model** on both training and testing data

---
| Dataset  | Accuracy                |
| -------- | ----------------------- |
| Training | \~69.38%                |
| Testing  | (Your test result here) |

📊 Future Improvements
Use other imputation strategies (median, most_frequent)

Try other classifiers (RandomForest, XGBoost)

Perform feature scaling (e.g., StandardScaler)

Tune hyperparameters using GridSearchCV



