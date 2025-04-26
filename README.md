# Classification Models Evaluation and Calibration

This repository contains experiments for evaluating multiple classification models on a dataset with class imbalance and calibration challenges. The workflow included data preparation, class imbalance handling, model training, calibration, and final model selection based on performance and calibration error.

---


## Data Preparation

- **Feature Removal**: Dropped four features that were entirely zeros, as they provided no discriminatory power.
- **Skewness Handling**: Checked numerical columns for skewness and applied log and power transformations to highly skewed ones. However, these transformations degraded model performance and were commented out.
- **Standardization**: All numerical data was standardized to ensure consistent scaling across features.


## Class Imbalance Handling

- **Neural Network**: Applied imbalance handling techniques such as **SMOTE** and **class weighting** — these significantly improved classification performance.
- **Classical ML Models**: Imbalance techniques worsened performance, particularly for the majority classes, so they were omitted.

---

## Models and Calibration Results

### Random Forest
| Metric                    | Before Calibration | After Calibration (Isotonic) |
|:-------------------------|:------------------|:----------------------------|
| Accuracy                  | 0.63               | 0.53                         |
| F1 (weighted avg)         | 0.60               | 0.52                         |
| ECE (Test Set)            | 0.1374             | 0.0973                       |
| ECE (Calibration Set)     | -                  | 0.2817                       |

**Notes**:
- Sigmoid calibration performed worse.
- Isotonic calibration reduced test set ECE without overfitting.


### Gradient Boosting
| Metric                    | Before Calibration | After Calibration (Isotonic) | After Calibration (Temp Scaling) |
|:-------------------------|:------------------|:----------------------------|:---------------------------------|
| Accuracy                  | 0.65               | 0.53                         | 0.65                            |
| F1 (weighted avg)         | 0.60               | 0.49                         | 0.60                            |
| ECE (Test Set)            | 0.1418             | 0.1812                       | 0.1770                          |
| ECE (Calibration Set)     | -                  | 0.2622                       | 0.1330                          |

**Notes**:
- Temperature scaling fit the calibration set better but resulted in a higher test set ECE.


### Logistic Regression
| Metric                    | Before Calibration | After Calibration (Isotonic) | After Calibration (Temp Scaling) |
|:-------------------------|:------------------|:----------------------------|:---------------------------------|
| Accuracy                  | 0.63               | 0.47                         | 0.63                            |
| F1 (weighted avg)         | 0.60               | 0.44                         | 0.60                            |
| ECE (Test Set)            | 0.1031             | 0.1430                       | 0.1100                          |
| ECE (Calibration Set)     | -                  | 0.1865                       | 0.1054                          |

**Notes**:
- Temperature scaling maintained accuracy and improved calibration on the calibration set.


### Neural Network
| Metric                    | Before Calibration | After Calibration (Temp Scaling) |
|:-------------------------|:------------------|:---------------------------------|
| Accuracy                  | 0.62               | 0.63                            |
| F1 (weighted avg)         | 0.63               | 0.63                            |
| ECE (Test Set)            | 0.2096             | 0.2910                          |
| ECE (Calibration Set)     | -                  | 0.2075                          |

**Notes**:
- Neural network exhibited high overconfidence.
- Calibration attempt with temperature scaling did not improve test set ECE.



## Final Model Selection

The **uncalibrated logistic regression model** was selected for deployment:
- Achieved the **lowest test set ECE** among all uncalibrated models.
- Offered a better balance between **classification performance** and **calibration error** than the calibrated random forest.

---

## Notebooks and Code

See the repository notebooks for:
- Data preprocessing details
- Model training and evaluation
- Calibration plots and ECE metrics
- Class imbalance experiments

