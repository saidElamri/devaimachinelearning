Churn Prediction Project

This project focuses on predicting customer churn for a telecom company using machine learning. The idea is to use past customer data (contracts, services, payments, etc.) to identify customers who are likely to leave, so the marketing team can take action earlier.

Project Overview

The work is divided into a few main parts:

Exploratory Data Analysis (EDA) in a Jupyter notebook to understand the data and spot patterns.

A machine learning pipeline that handles data preparation, model training, and evaluation.

A K-Fold cross-validation script to get a better estimate of model performance.

A simple model comparison between Logistic Regression, Random Forest, and SVC.

Some basic unit tests to check that everything runs as expected.

A short technical report to explain the approach and results.

How to Run

Clone the repo and install dependencies:

git clone https://github.com/saidElamri/devaimachinelearning
cd churn-prediction
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt


Run the notebook to explore the data:

jupyter notebook notebooks/eda_churn.ipynb


Train the model:

python src/pipeline.py


This will preprocess the data, train different models, compare them, and save the best model in the artifacts/ folder.

Run the K-Fold validation:

python src/kfold_validation.py


This script trains the model across several folds and prints the accuracy for each fold, along with the average accuracy.
You can easily switch between Logistic Regression, Random Forest, or SVC inside the file.

Run the tests:

pytest -q


The tests check things like data shapes and if the pipeline runs without errors.

Results

After testing different models, the one with the best balance between recall and F1-score was selected for the final pipeline.
The K-Fold validation gives a more reliable estimate of model performance compared to a single train/test split.

(Replace this section with your actual metrics and model choice.)

Next Steps

Deploy the model so it can be used in real campaigns.

Monitor its performance regularly.

Retrain it over time as customer behavior changes.

Folder Structure
.
├── notebooks/
│   └── EDA_Churn.ipynb
├── src/
│   ├── pipeline.py
│   └── kfold_validation.py
├── tests/
│   └── test_pipeline.py
├── artifacts/
├── reports/
│   └── rapport_technique.md
├── requirements.txt
└── README.md