Multi-Brand Marketing Campaign Performance Analysis and Prediction Using Python and Machine Learning

1\. Project Overview



This project focuses on analysing and comparing digital marketing campaign performance across three beauty and cosmetics brands: Nykaa, Purplle, and Tira.



The project uses Python, data analytics, visualization techniques, and machine learning to understand campaign performance, evaluate Return on Investment (ROI), analyse customer segments, and develop predictive models.



An interactive Streamlit dashboard is developed to present campaign insights and compare brand performance through visualizations and interactive filters.



2\. Project Objectives

Analyse marketing campaign performance across three brands.

Clean and preprocess marketing campaign datasets.

Perform Exploratory Data Analysis (EDA).

Identify campaign performance patterns using ROI.

Compare marketing channels, campaign types, and customer segments.

Develop a regression model for revenue prediction.

Develop a classification model for profitability prediction.

Evaluate machine learning models using appropriate performance metrics.

Build an interactive Streamlit dashboard.

3\. Brands Covered



The analysis includes:



Nykaa

Purplle

Tira

4\. Technologies Used

Technology	Purpose

Python	Programming

Pandas	Data manipulation and analysis

NumPy	Numerical operations

Matplotlib	Data visualization

Seaborn	Exploratory data analysis

Scikit-learn	Machine learning

Streamlit	Interactive dashboard

Jupyter Notebook	Data exploration and analysis

5\. Dataset Description



The project uses marketing campaign datasets containing information about:



Campaign ID

Campaign Type

Target Audience

Duration

Marketing Channels

Impressions

Clicks

Leads

Conversions

Revenue

Acquisition Cost

ROI

Language

Engagement Score

Customer Segment

Campaign Date

Brand



The combined original dataset contains 166,665 records.



After preprocessing and ROI-based filtering, the model-ready dataset contains 158,367 records.



6\. Project Methodology

Phase 1: Data Collection and Inspection

Collected datasets for Nykaa, Purplle, and Tira.

Combined the datasets.

Inspected columns, data types, missing values, and duplicates.

Performed initial data quality analysis.

Phase 2: Data Cleaning and Preprocessing

Handled missing categorical values.

Processed numerical and date fields.

Converted date information into useful features.

Prepared datasets for analysis and machine learning.

Phase 3: Exploratory Data Analysis



Performed analysis on:



Brand-wise ROI.

Campaign type performance.

Marketing channel performance.

Customer segment performance.

Target audience.

Language.

Numerical feature correlations.

ROI distribution.

Phase 4: Feature Engineering



Created additional features:



Campaign Quarter

Campaign Week

Click Through Rate

Conversion Rate

Lead Conversion Rate

Cost Per Click

Cost Per Lead

Revenue Per Conversion

Profit/Loss classification label



The Profit/Loss label is based on the supplied ROI:



Profit: ROI greater than zero.

Loss\_or\_BreakEven: ROI less than or equal to zero.

Phase 5: Revenue Prediction – Regression



Developed a Linear Regression model to predict campaign revenue.



Evaluation metrics:



Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Phase 6: Profit/Loss Prediction – Classification



Developed and evaluated:



Logistic Regression

Decision Tree Classifier



The target variable represents positive ROI versus non-positive ROI.



Phase 7: Model Evaluation



Compared model performance using appropriate evaluation metrics.



Phase 8: Interactive Dashboard



Developed a Streamlit dashboard with:



Campaign Overview

Brand Performance Comparison

Campaign Performance Analysis

ROI Analysis

Marketing Channel Analysis

Machine Learning Model Evaluation

Campaign Data Explorer



The dashboard includes interactive brand and campaign type filters and filtered dataset downloads.



7\. Machine Learning Results

Revenue Prediction – Linear Regression

Metric	Result

MAE	161,971.8326

RMSE	245,163.2306

R² Score	0.7413

Profit/Loss Classification

Model	Accuracy	Precision	Recall	F1 Score

Logistic Regression	0.4937	0.7593	0.4896	0.5954

Decision Tree	0.4038	0.7644	0.3127	0.4438



Logistic Regression was selected based on its higher F1 Score.



Note: The classification results indicate that there is scope for improving predictive performance. The classification target is based on ROI, not actual monetary profit.



8\. Project Structure

text

Marketing\_Campaign\_Predictions/

│

├── data/

│   ├── processed/

│   │   ├── combined\_dataset\_unmodified.csv

│   │   ├── cleaned\_dataset.csv

│   │   ├── model\_ready\_dataset.csv

│   │   └── feature\_engineered\_dataset.csv

│   │

│   └── raw/

│

├── dashboards/

│   └── app.py

│

├── models/

│   ├── best\_profitability\_model.pkl

│   └── best\_roi\_model.pkl

│

├── notebooks/

│

├── reports/

│   ├── brand\_roi\_analysis.csv

│   ├── campaign\_type\_roi\_analysis.csv

│   ├── channel\_used\_roi\_analysis.csv

│   ├── roi\_distribution.png

│   ├── roi\_boxplot.png

│   └── model\_evaluation\_summary.csv

│

├── src/

│   ├── data\_processing.py

│   ├── feature\_engineering.py

│   ├── revenue\_prediction.py

│   └── profitability\_classification.py

│

├── venv/

│

└── README.md



9\. Installation and Execution

Step 1: Clone the repository

bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL



Step 2: Navigate to the project

bash

cd Marketing\_Campaign\_Predictions



Step 3: Create a virtual environment

bash

python -m venv venv



Step 4: Activate the environment



Windows PowerShell:



powershell

.\\venv\\Scripts\\Activate.ps1



Step 5: Install dependencies

bash

pip install pandas numpy matplotlib seaborn scikit-learn streamlit



Step 6: Run the dashboard

bash

streamlit run dashboards/app.py





The dashboard will open in your web browser.



Note: The required processed dataset and evaluation report must be available in their expected project folders.



10\. Conclusion



This project demonstrates how Python-based data analytics and machine learning can be applied to marketing campaign performance analysis.



By analysing campaign data across Nykaa, Purplle, and Tira, the project provides insights into ROI, revenue, campaign types, marketing channels, and customer segments.



The developed regression and classification models demonstrate the application of machine learning to marketing analytics. The interactive Streamlit dashboard provides an accessible way to explore campaign performance and compare brands.



11\. Future Scope

Improve revenue prediction accuracy.

Explore additional machine learning algorithms.

Improve profitability classification performance.

Incorporate more marketing campaign datasets.

Add advanced interactive visualizations.

Deploy the dashboard online.



Project Domain: Marketing Analytics and Business Intelligence



Developed Using: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and Streamlit.

