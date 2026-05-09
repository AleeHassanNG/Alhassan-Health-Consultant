# Model Performance and Evaluation

This project uses supervised machine learning models to perform different healthcare-related predictive tasks.  
Each model was evaluated carefully using appropriate performance metrics to ensure that the models generalize properly and do not suffer from severe underfitting or overfitting.

---

# 1. Weight Assessment Model ⚖️

## Objective
The Weight Assessment module estimates a patient’s height based on weight measurements.

## Model Used
- **Simple Linear Regression**

## Features
- Input: Weight
- Target: Height

## Performance Metrics

| Metric | Training | Testing |
|---|---|---|
| Mean Squared Error (MSE) | 31.4669 | 34.3283 |
| Mean Absolute Error (MAE) | 4.4803 | 4.4125 |
| R² Score | 0.6700 | 0.5949 |

## Cross-Validation Results

| Metric | Value |
|---|---|
| Average Cross-Validated R² | 0.66996 |
| Standard Deviation | 0.00251 |

## Interpretation
The model explains approximately **59%–67%** of the variation in height using weight measurements.

The training and testing R² scores are reasonably close:

```text
Training R² = 0.6700
Testing R² = 0.5949
```

This suggests that the model is learning meaningful patterns without severe overfitting.

The cross-validation standard deviation is extremely small:

```text
0.00251
```

which indicates that the model behaves consistently across different subsets of the data.

Although the predictive strength is moderate, this is expected because height cannot be perfectly explained using weight alone.

---

# 2. BMI Evaluation Model 📊

## Objective
The BMI Evaluation module predicts Body Mass Index (BMI) using demographic and physical measurements.

## Model Used
- **Multiple Linear Regression**

## Features
- Height
- Weight
- Age
- Gender

## Target
- BMI

## Performance Metrics

| Metric | Training | Testing |
|---|---|---|
| Mean Squared Error (MSE) | 0.07589 | 0.07540 |
| Mean Absolute Error (MAE) | 0.20146 | 0.20109 |
| R² Score | 0.98422 | 0.98473 |

## Cross-Validation Results

| Metric | Value |
|---|---|
| Average Cross-Validated R² | 0.98421 |
| Standard Deviation | 0.00046 |

## Interpretation
This model achieved extremely strong predictive performance.

The R² scores are:

```text
Training R² = 0.98422
Testing R² = 0.98473
```

meaning that the model explains approximately **98.4%** of the variability in BMI.

The training and testing scores are almost identical, which strongly suggests that the model is:

- not underfitting
- not significantly overfitting
- generalizing very well to unseen data

The cross-validation results further confirm the stability of the model:

```text
Average R² = 0.98421
Standard Deviation = 0.00046
```

The extremely low standard deviation indicates very high consistency across folds.

This is expected because BMI is mathematically related to height and weight, making the relationship highly predictable.

---

# 3. Blood Pressure (BP) Risk Assessment ❤️

## Objective
The BP Risk Assessment module predicts blood pressure risk categories based on physiological and lifestyle-related variables.

## Model Used
- **Multiclass Logistic Regression**

## Features
- Diastolic Blood Pressure
- Systolic Blood Pressure
- BMI
- Smoking Status
- Activity Level
- Age
- Gender

## Target Classes
- Normal
- Elevated
- High

## Model Configuration

```python
LogisticRegression(
    C=0.1,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000
)
```

## Why Logistic Regression?
Logistic Regression was selected because:
- the target variable is categorical
- the problem involves multiclass classification
- the model is computationally efficient
- the model performs well on structured healthcare datasets

## Regularization and Overfitting Control
The BP model uses:

```text
L2 Regularization
```

with:

```text
C = 0.1
```

This helps reduce overfitting by penalizing excessively large model coefficients.

## Data Preprocessing
The BP model also included:
- feature standardization using `StandardScaler`
- outlier removal using the IQR method
- categorical encoding for smoking status, activity level, and gender

A total of:

```text
630 rows
```

were removed during outlier filtering to improve model stability.

## Interpretation
The BP Risk model was optimized for:
- stable classification
- efficient deployment in Streamlit
- multiclass health risk prediction

The use of regularization, preprocessing, and feature scaling improves generalization performance and reduces the likelihood of overfitting.

---

# Evaluation Metrics Explained

## R² Score
The R² score measures how much variability in the target variable is explained by the model.

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

- Values closer to 1 indicate stronger predictive performance.
- Values closer to 0 indicate weak predictive ability.

---

## Mean Squared Error (MSE)

MSE measures the average squared difference between predicted and actual values.

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

Lower values indicate better model performance.

---

## Mean Absolute Error (MAE)

MAE measures the average absolute prediction error.

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
\]

Smaller values indicate more accurate predictions.

---

# Overall Conclusion

The models implemented in this project demonstrated stable and reliable predictive performance.

- The **Weight Assessment model** showed moderate predictive capability appropriate for simple linear relationships.
- The **BMI Evaluation model** demonstrated excellent predictive accuracy with strong generalization ability.
- The **BP Risk Assessment model** effectively handled multiclass healthcare classification using Logistic Regression with regularization and preprocessing techniques.

Together, these models form a practical machine learning healthcare screening application suitable for:
- educational purposes
- research demonstrations
- introductory healthcare analytics
- predictive modeling practice using Streamlit and Scikit-Learn.
