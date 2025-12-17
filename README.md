# Predicting Wine Quality with Classical ML and Deep Learning  
**Team Members:**  
- Member 1 (Roll / ID)  
- Member 2 (Roll / ID)  
- Member 3 (Roll / ID)

---

## Abstract

This project predicts wine quality from physicochemical properties using both classical machine learning models and a deep learning approach. The original multi-class quality scores are imbalanced, so they are binned into three categories: *Poor*, *Average*, and *Good*. A Random Forest classifier, an SVM, and a fully connected neural network are trained and evaluated using accuracy and macro F1 score. Random Forest achieves the best macro F1 (≈0.655) and accuracy (≈0.698), while the deep learning model and SVM perform slightly worse but comparably. A McNemar test indicates that the performance difference between Random Forest and the deep model is not statistically significant at the 0.05 level. The findings highlight that ensemble classical models remain highly competitive for small, tabular datasets.

---

## Introduction

Wine quality assessment is important for wineries and distributors because it affects pricing, marketing strategy, and customer satisfaction. Traditionally, expert sommeliers provide quality ratings; however, this process is subjective, time-consuming, and hard to scale.  

The objective of this project is to build an automated system that predicts wine quality from laboratory measurements. Specifically, the project aims to:  
- Transform raw quality scores into more balanced, interpretable classes.  
- Compare classical ML algorithms with a deep learning model on the same tabular dataset.  
- Analyze performance using rigorous metrics and a statistical significance test.  
- Discuss the potential business value of deploying the best-performing model.

---

## Dataset Description

**Source**  
The dataset is the publicly available wine quality dataset (e.g., UCI Machine Learning Repository), containing physicochemical properties of wines and corresponding quality scores from human experts.

**Size and Target Transformation**  
- Total instances: original dataset of red (and/or white) wines.  
- Original target: integer quality scores (3–8), highly imbalanced with very few samples of 3, 4, and 8.  
- Target binning:  
  - 0 = *Poor*  
  - 1 = *Average*  
  - 2 = *Good*  
- Class distribution after binning:  
  - Poor (0): 522 samples  
  - Average (1): 462 samples  
  - Good (2): 159 samples  

**Features**  
Physicochemical attributes such as:  
- Fixed acidity, volatile acidity, citric acid  
- Residual sugar, chlorides  
- Free sulfur dioxide, total sulfur dioxide  
- Density, pH, sulphates, alcohol  

**Preprocessing**  
- Removal/merging of very rare quality classes via binning as described above.  
- Train/validation/test split:  
  - Train: 800 samples, 11 features  
  - Validation: 171 samples, 11 features  
  - Test: 172 samples, 11 features  
- Features kept in their original scales (or normalized/standardized if applied in code).  
- Handling of outliers (notably in residual sugar) was inspected via box plots but left in unless otherwise specified.  
- No missing values were present (or they were handled appropriately if found).

---

## Methodology

### Classical ML Approaches

Two main classical models were used:

1. **Random Forest Classifier**  
   - Ensemble of decision trees trained on bootstrapped samples.  
   - Tuned hyperparameters (best configuration found):  
     - `n_estimators = 200`  
     - `max_depth = 20`  
     - `min_samples_split = 2`  
     - `class_weight = 'balanced'`  

2. **Support Vector Machine (SVM)**  
   - Nonlinear kernel classifier targeting maximum margin separation.  
   - Tuned hyperparameters (best configuration found):  
     - `kernel = 'rbf'`  
     - `C = 1`  
     - `gamma = 'scale'`  
     - `class_weight = 'balanced'`  

Both models were trained on the training set and tuned using validation performance (or cross-validation over the training/validation split).

### Deep Learning Architectures

A fully connected feed-forward neural network was implemented:

- Input layer: 11 numeric features.  
- Hidden layers: 2–3 dense layers (e.g., 64–128 units each) with ReLU activations.  
- Output layer: 3 units with softmax activation (for the three classes).  
- Loss: categorical cross-entropy.  
- Optimizer: Adam (with default or tuned learning rate).  
- Regularization: dropout or L2 weight decay if applied.  

Training details:  
- Epochs: 50  
- Batch size: chosen based on dataset size (e.g., 32).  
- Training/validation loss tracked over epochs, showing gradual decrease in training loss and a validation loss curve that stabilizes and then slightly worsens (indicating mild overfitting).

### Hyperparameter Tuning Strategies

- **Classical models**  
  - Grid search or randomized search over candidate values of `n_estimators`, `max_depth`, `C`, `gamma`, and `class_weight`.  
  - Best model selected based on validation macro F1 score.  

- **Deep learning model**  
  - Manual tuning of network depth, hidden units, learning rate, and regularization.  
  - Early stopping considered via validation loss monitoring (best epoch chosen by validation loss).

---

## Results & Analysis

### Performance Comparison

Main evaluation metrics: **accuracy** and **macro F1 score** on the held-out test set.

#### Overall Metrics

| Model           | Accuracy  | Macro F1  |
|-----------------|-----------|----------:|
| Random Forest   | 0.6977    | 0.6552    |
| Deep Learning   | 0.6279    | 0.5973    |
| SVM             | 0.5988    | 0.5727    |

Random Forest obtains the highest macro F1 and accuracy, with the deep learning model and SVM trailing but not by a large margin.

#### Per-Class Metrics

**Random Forest (Test Set)**  

- Poor: precision 0.81, recall 0.73, F1 0.77 (support 79)  
- Average: precision 0.60, recall 0.75, F1 0.67 (support 69)  
- Good: precision 0.71, recall 0.42, F1 0.53 (support 24)  

**SVM (Test Set)**  

- Poor: precision 0.72, recall 0.72, F1 0.72 (support 79)  
- Average: precision 0.53, recall 0.45, F1 0.49 (support 69)  
- Good: precision 0.43, recall 0.62, F1 0.51 (support 24)  

**Deep Learning Model (Test Set)**  

- Poor: precision 0.66, recall 0.84, F1 0.74 (support 79)  
- Average: precision 0.63, recall 0.38, F1 0.47 (support 69)  
- Good: precision 0.52, recall 0.67, F1 0.58 (support 24)  

These results show that Random Forest and the deep model each have strengths: Random Forest balances performance across all classes, while the deep model performs particularly well on *Poor* and *Good* but struggles more on *Average*.

### Visualization of Results

The project includes several visualizations:

- Class distribution plots of raw quality scores and binned classes.  
- Feature correlation heatmap highlighting relationships between alcohol, density, acidity, and quality.  
- Box plot of residual sugar showing significant outliers.  
- Bar charts of macro F1 scores for Random Forest, SVM, and Deep Learning.  
- Confusion matrices for Random Forest and SVM, showing where misclassifications occur (especially between neighboring quality levels).

### Statistical Significance Tests

McNemar’s test was used to compare Random Forest and the deep learning model on the test set:

- Contingency Table (RF vs Deep): `[[91, 29], [17, 35]]`  
- p-value: `0.10381`  

At the 0.05 significance level, this p-value indicates **no statistically significant difference** between the two models. Although Random Forest scores higher on accuracy and macro F1, the improvement may be due to random variation given the sample size.

### Business Impact Analysis

If deployed in a winery or quality control pipeline, the best model could provide:

- **Consistent quality prediction** to support or complement human tasters.  
- **Cost savings** by detecting low-quality batches early in the production cycle.  
- **Pricing and segmentation support** through reliable classification into Poor/Average/Good tiers.  
- **Process improvement insights** by analyzing feature importance (e.g., impact of alcohol, acidity, or residual sugar on predicted quality).

Because the performance differences between Random Forest and the deep model are not statistically significant, model choice can also consider interpretability, training time, and ease of deployment.

---

## Conclusion & Future Work

This project demonstrates that classical ensemble methods, particularly Random Forest, are highly effective for predicting wine quality from physicochemical features on a small, tabular dataset. After binning imbalanced quality scores into three classes, the Random Forest model achieved the best macro F1 and accuracy among the tested methods, while the deep learning model and SVM performed slightly worse but similarly. McNemar’s test showed no statistically significant difference between Random Forest and the deep model, suggesting that practical considerations should guide final model selection.

**Future work** may include:

- Trying gradient boosting methods (XGBoost, LightGBM, CatBoost).  
- Exploring tabular-focused deep architectures and stronger regularization.  
- Using ordinal classification approaches to better respect the ordered nature of quality.  
- Collecting more data and using repeated cross-validation to obtain tighter performance and significance estimates.  
- Incorporating cost-sensitive learning where misclassifying high-quality wines carries greater penalty.

---

## References

1. UCI Machine Learning Repository – Wine Quality Data Set.  
2. L. Breiman, “Random Forests,” *Machine Learning*, 2001.  
3. C. Cortes and V. Vapnik, “Support-Vector Networks,” *Machine Learning*, 1995.  
4. D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” ICLR, 2015.  
5. Course notes and lecture slides provided by the instructor (evaluation metrics, McNemar’s test, and model selection guidelines).
