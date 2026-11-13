# 🩺 Disease Prediction from Medical Data | Machine Learning

A Machine Learning classification project that predicts the possibility of diseases using structured medical datasets.

Developed as part of my **Machine Learning Internship at CodeAlpha**.

## 📌 Project Overview

The Disease Prediction Model uses patient-related medical information to classify whether a disease is present or absent.

The project implements four Machine Learning classification algorithms and compares their performance across three medical datasets.

**Datasets included:**
- Breast Cancer
- Pima Indians Diabetes
- Heart Disease (Cleveland)

The positive class is represented as:

- `1` = Disease Present
- `0` = Disease Absent

The project focuses particularly on Recall because missing a disease-positive case (False Negative) can be costly in healthcare applications.

## 🎯 Project Objectives

- Predict disease possibilities using Machine Learning.
- Analyze structured medical datasets.
- Perform data preprocessing and feature engineering.
- Handle missing values using median imputation.
- Train and compare multiple classification algorithms.
- Evaluate model performance using classification metrics.
- Visualize ROC curves and feature importance.

## 🛠️ Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical computation |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning |
| XGBoost | Gradient boosting classification |

## 📊 Datasets Used

### 1. Breast Cancer Dataset
- Source: Scikit-learn built-in dataset.
- Contains breast cancer diagnostic measurements.
- Target: Malignant or Benign.
- Positive class: Malignant.

### 2. Diabetes Dataset
- Source: Pima Indians Diabetes Dataset.
- Contains medical measurements such as glucose, blood pressure, BMI, insulin, and age.
- Target: Diabetes outcome.

### 3. Heart Disease Dataset
- Source: Cleveland Heart Disease Dataset.
- Contains patient information such as age, cholesterol, resting blood pressure, maximum heart rate, and other clinical attributes.
- Target: Heart disease presence or absence.

## 🤖 Machine Learning Algorithms

This project implements four classification algorithms:

1. **Support Vector Machine (SVM)** – Identifies decision boundaries between disease-positive and disease-negative cases.
2. **Logistic Regression** – Estimates the probability of disease presence.
3. **Random Forest** – Combines multiple decision trees for classification.
4. **XGBoost** – Uses gradient boosting to build an ensemble classification model.

## ⚙️ Key Features

- Three medical datasets.
- Automated dataset loading.
- Data preprocessing and missing value handling.
- Feature engineering for Diabetes and Heart Disease datasets.
- StandardScaler for feature scaling.
- Class imbalance handling.
- Stratified train-test split (80:20).
- 5-fold Stratified Cross-Validation.
- Optional SVM hyperparameter tuning using GridSearchCV.
- Automatic model comparison.
- ROC curve visualization.
- Random Forest feature importance analysis.

## 📈 Model Evaluation Metrics

The models are evaluated using:

- **Accuracy:** Overall prediction correctness.
- **Precision:** Correct positive predictions among all positive predictions.
- **Recall:** Ability to identify actual disease-positive cases.
- **F1-Score:** Balance between Precision and Recall.
- **ROC-AUC:** Ability to distinguish between positive and negative classes.
- **Confusion Matrix:** Shows True Positives, True Negatives, False Positives, and False Negatives.

## 📁 Project Structure

```text
CodeAlpha_Disease_Prediction/
│
├── disease_prediction.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── results_breast_cancer.csv
├── results_diabetes.csv
├── results_heart.csv
│
├── plots_breast_cancer.png
├── plots_diabetes.png
└── plots_heart.png
```

The CSV result files and visualization images are generated automatically after running the program.

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/jayprakashbiswa49-pixel/CodeAlpha_Disease_Prediction.git
```

### 2. Navigate to the project directory

```bash
cd CodeAlpha_Disease_Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run all datasets

```bash
python disease_prediction.py
```

### 5. Run a specific dataset

For Breast Cancer:

```bash
python disease_prediction.py --dataset breast_cancer
```

For Diabetes:

```bash
python disease_prediction.py --dataset diabetes
```

For Heart Disease:

```bash
python disease_prediction.py --dataset heart
```

### 6. Enable SVM hyperparameter tuning

```bash
python disease_prediction.py --tune
```

## 📂 Using a Custom Dataset

You can also use your own CSV dataset:

```bash
python disease_prediction.py --csv my.csv --target label
```

**Important:**
- The target column must contain binary values.
- `1` must represent disease present.
- `0` must represent disease absent.
- Categorical features should be encoded before training.

## 📉 Generated Outputs

For each dataset, the program generates:

| Output | Description |
|---|---|
| `results_breast_cancer.csv` | Breast cancer model comparison |
| `results_diabetes.csv` | Diabetes model comparison |
| `results_heart.csv` | Heart disease model comparison |
| `plots_breast_cancer.png` | Breast cancer ROC and feature importance plots |
| `plots_diabetes.png` | Diabetes ROC and feature importance plots |
| `plots_heart.png` | Heart disease ROC and feature importance plots |

## 💡 Learning Outcomes

Through this project, I gained practical experience in:

- Supervised Machine Learning.
- Classification algorithms.
- Medical data preprocessing.
- Feature engineering.
- Handling class imbalance.
- Model comparison and evaluation.
- Cross-validation.
- Hyperparameter tuning.
- Data visualization using Matplotlib.
- Applying Machine Learning to healthcare-related problems.

## ⚠️ Disclaimer

This project is developed for educational and research purposes only.

The predictions are not medical diagnoses and should not be used for clinical decisions, treatment recommendations, or patient screening without extensive independent validation and appropriate medical oversight.

The datasets used may have limitations and may not represent all patient populations.

## 👨‍💻 Author

**Jayaprakash Biswal**

Machine Learning | Data Science | Python

- GitHub: [jayprakashbiswa49-pixel](https://github.com/jayprakashbiswa49-pixel)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/jayaprakash-biswal-bb1937414/)

## 🙌 Acknowledgment

Special thanks to **CodeAlpha** for providing the opportunity to learn and develop practical Machine Learning projects through the internship.

---

⭐ If you find this project interesting, consider giving this repository a star!