# Home Loan Default Prediction

## Overview:

This is a project created by Group 2 in the Bootcamp Data Analysis programe. 
This project aims to predict loan default using a dataset of loan application information. By analyzing various features related to applicants' personal, financial, and employment information, the goal is to build a machine learning model that can accurately predict whether an applicant is likely to default on their loan.
This repository contains code for predicting loan default using logistic regression. 

## Dataset:

The dataset used for this project is stored in the "application_train.csv" file. It contains 307,511 rows and 122 columns, including a mix of numerical and categorical variables. These variables capture information such as income, employment status, housing type, and credit history.

## Exploratory Data Analysis:
To explore the Data we have used Python, pandas, before building the predictive model, an exploratory data analysis (EDA) is performed to gain insights into the dataset and understand the relationships between different variables. Some key observations from the EDA include:

The target variable "TARGET" indicates whether an applicant has defaulted on their loan. The dataset is imbalanced, with a majority of non-defaulters (91.9%) compared to defaulters (8.1%).
Certain variables, such as "FLAG_OWN_CAR" and "FLAG_OWN_REALTY," show potential correlations with loan default.
Feature engineering is applied to create new variables representing loan amounts relative to the applicant's income.
Outliers in income and employment variables are handled appropriately.
We have also used Matplotlip to create visualization of the data, and use these visualizations for our presentation. 

## Data Preprocessing:

Before training the machine learning model, the dataset undergoes preprocessing steps, including:

Handling missing values: Missing values are identified and either dropped or imputed using suitable methods.
Encoding categorical variables: Categorical variables are encoded using techniques such as one-hot encoding or label encoding.
Feature scaling: Numerical variables are scaled to ensure uniformity and prevent any dominance of certain features.

Next, the code performs feature selection by computing the correlation matrix between the numerical features and the target variable. Features with a correlation value below the defined threshold (correlation_threshold = 0.1) are dropped from the dataset.

To handle categorical features, one-hot encoding is applied using the pd.get_dummies() function. This converts categorical columns into multiple binary columns, representing the presence or absence of each category.

The final dataset is formed by concatenating the one-hot encoded categorical features with the selected numerical features.

## Model Building and Evaluation:
 
The dataset is split into training and testing sets using the train_test_split() function from scikit-learn. The training set is further oversampled using the RandomOverSampler from the imbalanced-learn library to address class imbalance.
A logistic regression model is then instantiated and trained using the training data. The model is used to make predictions on the testing data, and accuracy, confusion matrix, and classification report are computed to evaluate the model's performance.
This ensures a balanced representation of defaulters and non-defaulters in both sets. The models are then trained on the training set and evaluated using appropriate performance metrics, such as accuracy, precision, recall, and F1 score.

## Results
The results obtained from the logistic regression model are as follows:

### Original Dataset:

* Accuracy: 91.93%
* Confusion Matrix:
  
![image](https://github.com/Misswieser/Project_4_Machine-Learning/assets/123671264/0b9dd387-6cd7-433e-86ca-3801f19cc206)

* Classification Report:

![image](https://github.com/Misswieser/Project_4_Machine-Learning/assets/123671264/6fdfa0a2-8cd3-4f33-a5ab-83c1ac174505)

### Oversampled Dataset:

* Accuracy: 66.70%
* Confusion Matrix:

![image](https://github.com/Misswieser/Project_4_Machine-Learning/assets/123671264/34a544b4-f3f0-4dc8-aeb5-b67a4c86d33b)

* Classification Report:

![image](https://github.com/Misswieser/Project_4_Machine-Learning/assets/123671264/c16dd1f4-a0b2-47e2-9df5-3bb090728f67)

The oversampling technique improves the recall of the minority class (loan default cases) but leads to a decrease in precision.

## Conclusion

In conclusion, this project focuses on predicting loan default using a dataset of loan application information. By leveraging machine learning techniques and appropriate data preprocessing steps, the project aims to build a predictive model that can assist in identifying the likelihood of loan default. This can provide valuable insights to financial institutions for making informed decisions and managing credit risk effectively.










