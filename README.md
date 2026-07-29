# Oncology Machine Learning Classifier

A predictive bioinformatics model that utilizes machine learning to classify breast cancer tumors as malignant or benign based on digitized image data of fine needle aspirates (FNA).

## Overview
This project demonstrates an end-to-end machine learning pipeline using Python and `scikit-learn`. The model is trained on the Wisconsin Breast Cancer dataset to analyze 30 continuous features of cell nuclei (such as radius, texture, perimeter, and area). It uses a Random Forest ensemble learning algorithm to achieve high diagnostic accuracy.

## Features
* **Data Splitting:** Implements `train_test_split` to prevent model overfitting.
* **Predictive Modeling:** Utilizes `RandomForestClassifier` for robust, multi-decision-tree predictive analytics.
* **Performance Evaluation:** Evaluates AI accuracy using statistical metrics including Precision, Recall, F1-Scores, and a generated Confusion Matrix.
* **Data Visualization:** Employs `seaborn` and `matplotlib` to render the model's prediction accuracy visually.

## Tech Stack
* **Language:** Python 3
* **Libraries:** `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

## How to Run
1. Clone this repository to your local machine.
2. Install the required dependencies from the requirements file:
   ```bash
   pip install -r requirements.txt

1. python cancer_classifier.py
