# American Express - Default Prediction

[Link to the competition](https://www.kaggle.com/competitions/amex-default-prediction)

As part of the competition, I ranked 627 out of 4876, placing in the top 13%. My performance was only 0.00217 points behind the top score and 0.014 points behind the fourth-place score.

## Summary of the Competition Activities

During the competition, the following tasks were accomplished:

1. **Time Series Analysis**: I studied time series data, which is crucial for understanding trends and making predictions over time.

2. **Gradient Boosting Models**: A large set of gradient boosting models were explored, including XGBoost, LightGBM, CatBoost, and more. This is detailed in the `amex-xgb-lgbm-catboost-cnn-stacking.ipynb` notebook.

3. **Stacking**: I gained experience in stacking, a technique that combines multiple machine learning models to improve prediction accuracy.

4. **Hyperparameters Tuning**: I honed my skills in hyperparameter tuning using cross-validation, a resampling procedure used to evaluate machine learning models on a limited data sample.

5. **Overfitting and Trust in Cross-Validation**: The competition provided a valuable lesson in overfitting to the leaderboard and the importance of trusting my cross-validation results. Had I trusted my cross-validation results, I would have ended up in the silver tier.

6. **Memory Management for Feature Generation**: I gained experience in managing operational memory during the feature generation process, which is important for handling large datasets and creating new features for model training.

7. **Feature Engineering and Feature Selection**: I acquired experience in feature engineering and feature selection, which are critical steps in the machine learning pipeline to improve the performance of models.

These experiences have significantly enriched my skills in machine learning and data science, and I'm looking forward to applying these learnings to future competitions.


<details>
<summary><h1>Brute Force Feature Engineering for AMEX Dataset</h1></summary>
This repository contains a Jupyter notebook that demonstrates a brute force method for feature engineering on the AMEX dataset.

**You can run: amex-extra_bruteforce-feature-engineering.ipynb**

## Overview
The approach builds on the ideas presented in two high-scoring notebooks:

1. "Amex LGBM Dart CV 0.7977" by Martin Kovacevic Buvinic, which introduced features based on the differences between the last value and lag_1, and the last value and the average.
2. "Lag Features Are All You Need" which introduced features based on the first value, first and last interactions, and combined them into a single highest scoring notebook.

This notebook takes the concept further by computing features based on interactions with the last feature for all columns in the dataset.

## Features
The notebook introduces two new types of features for each column:

- Last - col: The difference between the column and the last value of this column.
- Last / col: The fractional difference between the column and the last value of this column.

## Dataset
The dataset used in this notebook has been preprocessed and extracted for public use. The link to the dataset will be provided soon.

## Usage
To use this notebook, simply open it in Jupyter and run all the cells. The feature engineering is done in a step-by-step manner, and the notebook should be easy to follow.

## Results
Applying this brute force feature engineering method yielded an improvement in the model's score.

## Additional Feature Engineering
In addition to the features based on interactions with the last value, this notebook also calculates a range of statistical features for each column, providing a deeper understanding of the data distribution. These features include:

- Minimum Value (min_value): The smallest value in each column.
- Maximum Value (max_value): The largest value in each column.
- Mean Value (mean_value): The average value of each column.
- Median Value (median_value): The middle value of each column.
- Mode Value (mode_value): The most frequently occurring value in each column.
- Standard Deviation (std_dev): The amount of variation in each column.
- Variance (variance): The squared standard deviation for each column.
- Sum of Values (sum_values): The total sum of all values in each column.
- Product of Values (product_values): The result of multiplying all the values in each column.
- First Quartile (first_quartile): The value below which a quarter of the data falls.
- Third Quartile (third_quartile): The value below which three quarters of the data falls.
- 1st Percentile (percentile_1): The value below which 1% of the data falls.
- 5th Percentile (percentile_5): The value below which 5% of the data falls.
- 95th Percentile (percentile_95): The value below which 95% of the data falls.
- 99th Percentile (percentile_99): The value below which 99% of the data falls.
- First Value (first_value): The first value of each column.
- Last Value (last_value): The last value of each column.
- Count of Values (count_values): The number of values in each column.
- Norm Value (norm_value): The Euclidean norm (or magnitude) of the values in each column.

These functions are defined using numpy and scipy libraries, and are applied to each column in the dataset. The resulting features provide a comprehensive summary of the data, enhancing the model's ability to capture complex patterns and relationships.

</details>

<details>
<summary><h1>Hyperparameters Experiment for AMEX Dataset</h1></summary>
This repository contains a Jupyter notebook that explores various hyperparameters for a model trained on the AMEX dataset.

**You can run: amex-hyperparameters_experiment.ipynb**

## Overview

Hyperparameters are crucial in defining the model structure and controlling the learning process. In this notebook, we experiment with different hyperparameter settings to understand their impact on model performance.

## Hyperparameters

Hyperparameters in machine learning models include settings like learning rate, number of hidden layers (for neural networks), or number of trees (for tree-based methods like Random Forests or Gradient Boosting), among others.

## Experiment Process

The notebook systematically varies these hyperparameters, trains a model with each combination, and measures the resulting model's performance. By doing this, we can empirically identify which hyperparameters lead to the best performance on our dataset.

## Dataset

The dataset used in this notebook is the AMEX dataset, preprocessed and extracted for public use.

## Usage

To use this notebook, open it in Jupyter and run all cells. The process of hyperparameter experimentation is executed in a step-by-step manner, making the notebook easy to follow.

## Results

Through this hyperparameter tuning process, we aim to optimize the model's performance. The results section in the notebook provides a detailed analysis of the best performing hyperparameters.

## Note

Hyperparameter tuning is a broad and deep field, and the settings that work best can depend heavily on the specifics of the dataset and the model being used. Therefore, the hyperparameters identified in this notebook are specific to the AMEX dataset and the particular model used.

</details>

<details>
<summary><h1>Stacked Machine Learning Model</h1></summary>

This notebook contains a machine learning model that combines the predictions of XGBoost, LightGBM, CatBoost, and a Convolutional Neural Network (CNN) for a binary classification problem.

**You can run: anex-xgb-lgbm-catboost-cnn-stacking.ipynb**

## Structure

1. **Imports**: Libraries needed for data manipulation, model creation, and evaluation.
2. **Preprocessing Functions**: Functions to perform preprocessing and feature engineering on the dataset.
3. **Data Loading Functions**: Functions to load the train and test data.
4. **Model Training Functions**: Functions to train the XGBoost, LightGBM, CatBoost, and CNN models.
5. **Metric Functions**: Functions to calculate a custom evaluation metric.
6. **Model Training**: The main section where the models are trained using 5-fold cross-validation and predictions are made.
7. **Test Prediction**: Predictions are made on the test set using the trained models.
8. **Submission**: The final predictions are saved in a CSV file for submission.

## Usage

This notebook is designed to be run in a Jupyter environment. Ensure that you have the required libraries installed and the dataset accessible in the specified path. Update the path as needed to point to your dataset.

## Note

This model uses a custom metric for evaluation and specific preprocessing steps. Please understand these before using the model.
</details>