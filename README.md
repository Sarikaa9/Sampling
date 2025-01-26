# Sampling and Machine Learning Analysis
This repository contains Python code for analyzing different sampling techniques and their impact on the performance of various machine learning models.

## Dataset
The dataset used in this analysis is a credit card fraud dataset, downloaded directly from the GitHub link:
[Credit Card Dataset](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv).

## Objective
The primary objectives of this project are:
1. Balance the dataset classes using SMOTE (Synthetic Minority Over-sampling Technique).
2. Create five samples of the dataset based on a sample size detection formula.
3. Apply five different sampling techniques:
   - Sampling1 (No sampling)
   - Sampling2 (Random undersampling)
   - Sampling3 (SMOTE)
   - Sampling4 (SMOTEENN)
   - Sampling5 (Custom 80% random sampling)
4. Evaluate the performance of five machine learning models:
   - M1: Random Forest Classifier
   - M2: Logistic Regression
   - M3: Decision Tree Classifier
   - M4: Support Vector Machine (SVC)
   - M5: K-Nearest Neighbors (KNN)

## Results
The code determines:
- The accuracy of each model under different sampling techniques.
- The best sampling technique for each model along with its accuracy.

## How to Run
1. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn
