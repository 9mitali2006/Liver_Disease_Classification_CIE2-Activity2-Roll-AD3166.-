# 🏥 LiverCare AI – Liver Disease Classification Using Logistic Regression

## 1. Project Overview

LiverCare AI is a Machine Learning project that uses **Logistic Regression** to classify whether a person belongs to the Liver Disease patient group or the Non-Liver Disease patient group based on medical attributes.

The project uses the Indian Liver Patient Dataset and demonstrates how Machine Learning can be applied to binary classification problems.

Two Logistic Regression models are implemented:

1. **Normal Logistic Regression:** A standard Logistic Regression model without class weighting.
2. **Balanced Logistic Regression:** A Logistic Regression model using `class_weight="balanced"` to handle class imbalance.

The project also explores important Machine Learning concepts such as:
- Binary Classification
- Sigmoid Function
- Probability Prediction
- Decision Threshold
- Feature Scaling
- Missing Value Imputation
- Class Imbalance
- Confusion Matrix
- Classification Metrics

## 2. Problem Statement

To develop a Machine Learning classification model that predicts whether a person belongs to the Liver Disease or Non-Liver Disease category based on medical attributes.

The objective is to understand how Logistic Regression performs on medical classification data and how techniques such as class weighting and threshold adjustment can influence classification results.

## 3. Project Objectives

- Understand binary classification using Machine Learning.
- Explore and analyze the Indian Liver Patient Dataset.
- Perform data preprocessing.
- Handle missing values using median imputation.
- Convert categorical data into numerical format.
- Apply feature scaling using StandardScaler.
- Understand and implement the Sigmoid Function.
- Build a Logistic Regression classification model.
- Compare Normal and Balanced Logistic Regression.
- Experiment with different classification thresholds.
- Evaluate model performance using classification metrics.
- Analyze confusion matrices and classification errors.

## 4. Dataset Description

**Dataset Name:** Indian Liver Patient Dataset

**Total Records:** 583

**Total Attributes:** 11

The dataset contains medical information about individuals, including age, gender, bilirubin levels, enzyme levels, proteins, and albumin measurements.

### Dataset Attributes

| Attribute | Description |
|---|---|
| Age | Age of the individual |
| Gender | Gender of the individual |
| Total_Bilirubin | Total bilirubin level |
| Direct_Bilirubin | Direct bilirubin level |
| Alkaline_Phosphotase | Alkaline phosphatase level |
| Alamine_Aminotransferase | ALT enzyme level |
| Aspartate_Aminotransferase | AST enzyme level |
| Total_Protiens | Total protein level |
| Albumin | Albumin level |
| Albumin_and_Globulin_Ratio | Albumin to globulin ratio |
| Dataset | Target classification |

### Target Variable

The target variable is `Dataset`.

The original dataset uses the following values:

- 1 = Liver Disease
- 2 = Non-Liver Disease

For this project, the target variable is converted into binary format:

- 1 = Liver Disease
- 0 = Non-Liver Disease

### Input Features

The following 10 features are used for prediction:

1. Age
2. Gender
3. Total_Bilirubin
4. Direct_Bilirubin
5. Alkaline_Phosphotase
6. Alamine_Aminotransferase
7. Aspartate_Aminotransferase
8. Total_Protiens
9. Albumin
10. Albumin_and_Globulin_Ratio

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning implementation |
| Google Colab / Jupyter Notebook | Development environment |

## 6. Project Workflow

The project follows these steps:

1. Import required Python libraries.
2. Load the Indian Liver Patient Dataset.
3. Explore dataset structure and attributes.
4. Check missing values.
5. Convert Gender into numerical format.
6. Convert the target variable into binary format.
7. Separate independent and dependent variables.
8. Split the dataset into training and testing sets.
9. Understand the Sigmoid Function.
10. Create a Machine Learning pipeline.
11. Handle missing values using SimpleImputer.
12. Perform feature scaling using StandardScaler.
13. Train Normal Logistic Regression.
14. Generate probability predictions.
15. Apply a classification threshold.
16. Evaluate the normal model.
17. Train Balanced Logistic Regression.
18. Compare both models.
19. Experiment with different decision thresholds.
20. Visualize and compare confusion matrices.

## 7. Data Preprocessing

Data preprocessing is performed to prepare the dataset for Machine Learning.

### 7.1 Gender Encoding

The Gender column contains categorical values:

- Male
- Female

Machine Learning algorithms require numerical input, so the values are converted as follows:

| Gender | Encoded Value |
|---|---:|
| Male | 1 |
| Female | 0 |

### 7.2 Target Variable Encoding

The original Dataset column contains:

- 1 = Liver Disease
- 2 = Non-Liver Disease

It is converted into:

- 1 = Liver Disease
- 0 = Non-Liver Disease

This converts the problem into binary classification.

### 7.3 Missing Value Handling

The dataset contains 4 missing values in the `Albumin_and_Globulin_Ratio` column.

The `SimpleImputer(strategy="median")` technique is used to handle missing values.

**Why Median Imputation?**

Median replaces missing values with the middle value of the available observations.

It is useful when data may contain extreme values because the median is less affected by outliers than the mean.

### 7.4 Feature Scaling

The `StandardScaler()` technique is used to standardize numerical features.

Standardization transforms the features so that they have approximately:

- Mean = 0
- Standard Deviation = 1

**Formula:**

`Z = (X - Mean) / Standard Deviation`

**Why Feature Scaling?**

The dataset contains features with different measurement ranges.

For example:
- Age may range between different age values.
- Bilirubin levels have smaller numerical values.
- Enzyme levels may have much larger numerical values.

Scaling ensures that features with larger numerical ranges do not dominate the optimization process simply because of their magnitude.

## 8. Train-Test Split

The dataset is divided into training and testing datasets using `train_test_split()`.

### Configuration

| Parameter | Value |
|---|---|
| Test Size | 20% |
| Training Size | 80% |
| Random State | 42 |
| Stratify | y |

### Dataset Distribution

| Dataset | Records |
|---|---:|
| Total Dataset | 583 |
| Training Dataset | 466 |
| Testing Dataset | 117 |

### Why Stratified Sampling?

The `stratify=y` parameter maintains approximately the same class distribution in both training and testing datasets.

This is especially useful when the dataset contains an unequal number of observations in different classes.

## 9. Understanding Logistic Regression

Logistic Regression is a supervised Machine Learning algorithm used for classification problems.

Although its name contains the word Regression, it is primarily used to predict categories.

In this project, the two categories are:

- Liver Disease
- Non-Liver Disease

Unlike Linear Regression, which predicts continuous numerical values, Logistic Regression predicts the probability that an observation belongs to a particular class.

### 9.1 Sigmoid Function

The Sigmoid Function converts any real-valued input into a probability between 0 and 1.

**Formula:**

`σ(z) = 1 / (1 + e^(-z))`

Where:

- z = Linear combination of input features and coefficients
- e = Euler's number
- σ(z) = Predicted probability

### Why Sigmoid Function?

The output of a mathematical equation can be any real number.

However, classification requires a probability between 0 and 1.

The Sigmoid Function solves this problem by converting the output into a probability.

### 9.2 Decision Threshold

The decision threshold determines which class should be assigned based on the predicted probability.

The default threshold used in this project is:

`Threshold = 0.5`

Classification rules:

- Probability >= 0.5 → Liver Disease (1)
- Probability < 0.5 → Non-Liver Disease (0)

The project also experiments with different thresholds to understand their impact on classification performance.

## 10. Model Implementation

Two Logistic Regression models are developed.

### 10.1 Normal Logistic Regression

The first model uses standard Logistic Regression without applying class weighting.

**Configuration:**

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Max Iterations | 1000 |
| Class Weight | None |
| C | 1.0 |
| Decision Threshold | 0.5 |

The model is implemented using a Scikit-learn Pipeline.

The pipeline contains three steps:

1. SimpleImputer – Handles missing values using the median.
2. StandardScaler – Standardizes input features.
3. LogisticRegression – Performs binary classification.

The model learns patterns from the training data and predicts the probability of Liver Disease for the testing data.

### 10.2 Balanced Logistic Regression

The second model uses Logistic Regression with:

`class_weight="balanced"`

### Why Balanced Logistic Regression?

The dataset contains an unequal distribution of Liver Disease and Non-Liver Disease observations.

When a dataset is imbalanced, a model may perform well on the majority class but poorly on the minority class.

Balanced Logistic Regression automatically adjusts class weights based on the frequency of each class.

This gives more importance to the minority class during model training.

**Configuration:**

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Max Iterations | 1000 |
| Class Weight | balanced |
| C | 1.0 |
| Decision Threshold | 0.5 |

The same preprocessing pipeline is used for both models to ensure a fair comparison.

## 11. Model Evaluation Metrics

The following evaluation metrics are used to measure model performance.

### 11.1 Accuracy

Accuracy measures the percentage of correctly classified observations.

**Formula:**

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

Higher accuracy indicates that more predictions are correct.

However, accuracy alone may not provide a complete picture when the dataset is imbalanced.

### 11.2 Precision

Precision measures how many observations predicted as positive actually belong to the positive class.

**Formula:**

`Precision = TP / (TP + FP)`

In this project, it measures how many predicted Liver Disease cases are actually Liver Disease cases.

### 11.3 Recall

Recall measures how many actual positive observations are correctly identified by the model.

**Formula:**

`Recall = TP / (TP + FN)`

In this project, it measures how many actual Liver Disease cases are correctly identified.

### 11.4 F1-Score

F1-Score is the harmonic mean of Precision and Recall.

**Formula:**

`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

It provides a combined measure of precision and recall.

### 11.5 Log Loss

Log Loss evaluates the quality of predicted probabilities.

It penalizes predictions that are confidently incorrect.

A lower Log Loss indicates better probability prediction performance.

## 12. Confusion Matrix

A Confusion Matrix is used to understand the classification results in detail.

It contains four values:

| Term | Meaning |
|---|---|
| True Positive (TP) | Actual Liver cases correctly predicted as Liver |
| True Negative (TN) | Actual Non-Liver cases correctly predicted as Non-Liver |
| False Positive (FP) | Actual Non-Liver cases incorrectly predicted as Liver |
| False Negative (FN) | Actual Liver cases incorrectly predicted as Non-Liver |

The confusion matrix helps identify which types of classification errors the model is making.

## 13. Results and Model Comparison

Both models are evaluated using the same testing dataset of 117 records.

### 13.1 Normal Logistic Regression Results

**Accuracy:** 73.50%

**Log Loss:** 0.4530

Confusion Matrix:

| | Predicted Non-Liver | Predicted Liver |
|---|---:|---:|
| Actual Non-Liver | 6 | 28 |
| Actual Liver | 3 | 80 |

Classification Report:

| Metric | Non-Liver | Liver |
|---|---:|---:|
| Precision | 0.67 | 0.74 |
| Recall | 0.18 | 0.96 |
| F1-Score | 0.28 | 0.84 |
| Support | 34 | 83 |

**Observation:**

The normal Logistic Regression model correctly identifies most Liver Disease cases.

However, it struggles to identify Non-Liver Disease cases, achieving only 17.65% recall for that class.

### 13.2 Balanced Logistic Regression Results

**Accuracy:** 73.50%

Confusion Matrix:

| | Predicted Non-Liver | Predicted Liver |
|---|---:|---:|
| Actual Non-Liver | 29 | 5 |
| Actual Liver | 26 | 57 |

Classification Report:

| Metric | Non-Liver | Liver |
|---|---:|---:|
| Precision | 0.53 | 0.92 |
| Recall | 0.85 | 0.69 |
| F1-Score | 0.65 | 0.79 |
| Support | 34 | 83 |

**Observation:**

Balanced Logistic Regression improves Non-Liver recall significantly.

It increases Non-Liver recall from 17.65% to 85.29%.

However, Liver Disease recall decreases from 96.39% to 68.67%.

This shows that class weighting changes how the model handles the two classes.

### 13.3 Model Comparison

| Metric | Normal Logistic Regression | Balanced Logistic Regression |
|---|---:|---:|
| Accuracy | 73.50% | 73.50% |
| Non-Liver Precision | 0.67 | 0.53 |
| Non-Liver Recall | 0.18 | 0.85 |
| Liver Precision | 0.74 | 0.92 |
| Liver Recall | 0.96 | 0.69 |
| Non-Liver F1-Score | 0.28 | 0.65 |
| Liver F1-Score | 0.84 | 0.79 |

### Key Findings

- Both models achieve approximately 73.50% accuracy.
- Normal Logistic Regression identifies more Liver Disease cases correctly.
- Balanced Logistic Regression improves the identification of Non-Liver Disease cases.
- Class weighting changes the balance between the two classes.
- Accuracy remains unchanged, but class-wise performance changes significantly.

## 14. Decision Threshold Experiment

The project experiments with five different decision thresholds using the Balanced Logistic Regression model.

The thresholds are:

- 0.3
- 0.4
- 0.5
- 0.6
- 0.7

### Results

| Threshold | Accuracy | True Negative | False Positive | False Negative | True Positive |
|---|---:|---:|---:|---:|---:|
| 0.3 | 0.744 | 9 | 25 | 5 | 78 |
| 0.4 | 0.761 | 21 | 13 | 15 | 68 |
| 0.5 | 0.735 | 29 | 5 | 26 | 57 |
| 0.6 | 0.667 | 33 | 1 | 38 | 45 |
| 0.7 | 0.615 | 34 | 0 | 45 | 38 |

### Observations

**Threshold 0.3:**
- Identifies more Liver Disease cases.
- Has fewer False Negatives.
- Produces more False Positives.

**Threshold 0.4:**
- Achieves the highest accuracy among the tested thresholds.
- Provides a different balance between False Positives and False Negatives.

**Threshold 0.5:**
- Provides the default classification threshold.
- Produces 29 True Negatives and 57 True Positives.

**Threshold 0.6:**
- Reduces False Positives.
- Increases False Negatives.

**Threshold 0.7:**
- Produces zero False Positives.
- However, it misses many actual Liver Disease cases.

### Conclusion of Threshold Experiment

Changing the decision threshold changes how the model classifies observations.

A lower threshold makes the model more likely to predict Liver Disease, while a higher threshold makes it more conservative.

The appropriate threshold depends on the objective and the relative cost of different classification errors.

## 15. Confusion Matrix Visualization

The project visualizes two confusion matrices side by side:

1. Normal Logistic Regression
2. Balanced Logistic Regression

The `ConfusionMatrixDisplay` function from Scikit-learn is used.

### Purpose of Visualization

The visualization helps compare:

- Correctly classified Liver Disease cases.
- Correctly classified Non-Liver Disease cases.
- False Positive predictions.
- False Negative predictions.

It makes the differences between the two models easier to understand.

## 16. Important Machine Learning Concepts Used

| Concept | Purpose |
|---|---|
| Binary Classification | Classifies observations into two categories |
| Logistic Regression | Predicts the probability of a class |
| Sigmoid Function | Converts model output into probability |
| Decision Threshold | Converts probability into a class |
| SimpleImputer | Handles missing values |
| StandardScaler | Standardizes numerical features |
| Train-Test Split | Separates training and testing data |
| Stratified Sampling | Maintains class distribution |
| Class Weighting | Handles class imbalance |
| Confusion Matrix | Analyzes classification errors |
| Accuracy | Measures overall correctness |
| Precision | Measures positive prediction quality |
| Recall | Measures identification of actual positive cases |
| F1-Score | Balances precision and recall |
| Log Loss | Evaluates predicted probabilities |

## 17. How to Run the Project

### Prerequisites

Install Python and the required libraries.

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
```

### Step 2: Navigate to the Project Folder

```bash
cd LiverCare-AI
```

### Step 3: Add the Dataset

Download the Indian Liver Patient Dataset and place the CSV file inside the project directory.

Dataset filename used in the notebook:

`indian_liver_patient.csv`

### Step 4: Update the Dataset Path

Update the dataset loading code in the notebook:

```python
df = pd.read_csv("indian_liver_patient.csv")
```

### Step 5: Open Jupyter Notebook

```bash
jupyter notebook
```

Alternatively, upload the notebook to Google Colab.

### Step 6: Run the Notebook

Execute all cells sequentially to perform:

- Dataset exploration
- Data preprocessing
- Sigmoid visualization
- Logistic Regression training
- Balanced Logistic Regression training
- Model evaluation
- Threshold experimentation
- Confusion matrix visualization

## 18. Project Structure

```text
LiverCare-AI/
│
├── RCA_CIE2_ACT2(3).ipynb
│
├── indian_liver_patient.csv
│
└── README.md
```

## 19. Conclusion

This project demonstrates the implementation of Logistic Regression for binary classification using the Indian Liver Patient Dataset.

Two models are developed and evaluated:

1. Normal Logistic Regression
2. Balanced Logistic Regression

Both models achieve approximately 73.50% accuracy.

However, their class-wise performance is different.

Normal Logistic Regression achieves better Liver Disease recall, while Balanced Logistic Regression significantly improves Non-Liver Disease recall.

The project also demonstrates how changing the classification threshold influences True