# Diabetes Prediction Project

## Introduction

Diabetes is a widespread chronic condition that can lead to severe health complications if not managed effectively. Early prediction and diagnosis of diabetes play a crucial role in preventing and mitigating these complications. In this project, we aim to develop a machine learning model to predict the likelihood of diabetes in individuals based on various health indicators.

## Project Overview

This machine learning project focuses on building a predictive model for diabetes using a logistic regression algorithm. The model is trained on a dataset containing several health-related features that are known to influence diabetes risk. The main goal is to assess the model’s ability to predict diabetes with high accuracy and precision, which could aid healthcare professionals in early detection and personalized treatment planning.

## Objectives

- To preprocess and analyze the dataset to understand the key health indicators associated with diabetes.
- To develop a predictive model using logistic regression for classifying individuals as diabetic or non-diabetic.
- To evaluate the model's performance on training and testing data to determine its effectiveness in predicting diabetes.

## Dataset Overview

The dataset consists of records with various health metrics relevant to diabetes risk. The target variable is whether the individual has diabetes, based on the following features:

### Key Features (Columns)
- **gender**: Gender of the individual.
- **age**: Age of the individual.
- **hypertension**: Presence (1) or absence (0) of hypertension.
- **heart_disease**: Presence (1) or absence (0) of heart disease.
- **smoking_history**: History of smoking behavior.
- **bmi**: Body Mass Index, a measure of body fat based on height and weight.
- **HbA1c_level**: Average blood glucose levels over the past 3 months.
- **blood_glucose_level**: Current blood glucose level.
- **diabetes**: Target variable, indicating if the individual has diabetes (1) or not (0).

## Dataset Analysis

The dataset undergoes a series of preprocessing steps, including:
- **Handling missing values**: Ensuring data completeness for each feature.
- **Data scaling**: Normalizing certain features to improve model performance.
- **Feature selection**: Retaining features that have a significant impact on diabetes prediction.

Exploratory Data Analysis (EDA) is conducted to observe distributions, detect correlations, and understand the relationships between features and the target variable, which helps in feature selection and model tuning.

## Model Selection

### Algorithm Used

After analyzing various machine learning algorithms, **Logistic Regression** was chosen for this project due to its effectiveness in binary classification problems. Logistic regression is simple, interpretable, and performs well with linear relationships.

### Training Process

The model training process involved:
1. **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets to evaluate model performance.
2. **Model Training**: Logistic regression was trained on the training set.
3. **Evaluation**: The model’s accuracy and precision were evaluated on both training and testing datasets to ensure consistency and reliability.

## Results

### Model Performance

- **Data Split**: 80% training data, 20% testing data
- **Accuracy**:
  - Training data: 96.01%
  - Testing data: 95.77%
- **Precision**:
  - Training data: 59.48%
  - Testing data: 57.37%

### Conclusion

The logistic regression model demonstrates strong accuracy, indicating that it performs well in predicting diabetes on the given dataset. While precision shows moderate values, this can be improved with further hyperparameter tuning and potentially additional features.

This model can serve as a helpful tool in the early detection of diabetes, offering valuable insights into patient risk profiles based on common health indicators.
