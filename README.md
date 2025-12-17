# Wine Quality Classification Project

## 1. Project Overview & Objectives
**Course:** CS 470 - Machine Learning
**Objective:** Compare Classical ML (RF, SVM) vs Deep Learning (ResNet) on WineQT dataset.

### Abstract
We formulated the problem as a multi-class classification task (Poor, Average, Good). Our results indicate that **Random Forest achieved the highest Macro F1-Score of 0.655**, demonstrating that ensemble tree methods often outperform neural networks on small tabular datasets.

## 2. Methodology
* **Preprocessing:** Target binning (0: Poor, 1: Average, 2: Good), SMOTE strategy via Class Weights.
* **Classical Models:** Random Forest (GridSearch), SVM (RBF Kernel).
* **Deep Learning:** Residual MLP with Batch Norm & Dropout (0.3).

## 3. Results Table

| Model | Macro F1-Score |
|-------|----------------|
| **Random Forest** | **0.655** |
| SVM | 0.573 |
| Deep Learning | 0.597 |

## 4. Business Impact Analysis
This project offers significant value to the wine industry by automating quality control:
* **Efficiency:** The model automates preliminary grading, processing batches 100x faster than manual tasting.
* **Consistency:** It eliminates human bias and fatigue, ensuring standardized quality ratings across different production batches.
* **Cost Reduction:** By filtering out "Poor" and "Average" wines automatically, vineyards can reserve expensive expert sommeliers only for the "Good" rated batches.

## 4. Conclusion
Random Forest is recommended for deployment due to superior F1 performance and training efficiency on this dataset size.
