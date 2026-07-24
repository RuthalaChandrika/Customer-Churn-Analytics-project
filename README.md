📊 Telecom Customer Churn Analysis & Prediction
End-to-End Data Analytics & Machine Learning Project using Python, EDA, Random Forest, SMOTE, and Power BI

📌 Project Overview
Customer churn is one of the biggest challenges faced by telecom companies. Losing customers directly impacts revenue and business growth.
This project analyzes customer behavior, identifies key factors influencing churn, and builds a machine learning model to predict customers
who are likely to leave the service.

The solution combines:
Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Business Insights Generation
Machine Learning Model Development
Interactive Power BI Dashboard

🎯 Business Problem
A telecom company is experiencing a high number of customer cancellations (churn).
The company wants to:
Understand why customers leave
Identify high-risk customers
Reduce churn through proactive retention strategies
Improve customer satisfaction and revenue

🎯 Project Objectives
✔ Analyze customer demographics and service usage
✔ Identify major factors contributing to churn
✔ Perform Exploratory Data Analysis (EDA)
✔ Build a machine learning model for churn prediction
✔ Create a Power BI dashboard for business user
✔ Provide actionable recommendations for customer retention

❓ Business Questions Solved
Which customers are more likely to churn?
Does contract type affect churn?
How do monthly charges influence churn?
Does online security reduce churn?
Does premium tech support improve retention?
Which payment methods have higher churn rates?
Are high internet/data users more likely to leave?
Which offers improve customer retention?

🛠 Technologies Used
Programming
Python
Libraries
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Imbalanced-Learn (SMOTE)
Joblib
Business Intelligence
Power BI
Machine Learning
Random Forest Classifier
SMOTE (Class Imbalance Handling)

📂 Dataset Information:
The dataset contains 7,043 telecom customers and includes:
Customer Information
Gender
Age
Marital Status
Dependents
City
Population
Service Information
Internet Service
Internet Type
Phone Service
Online Security
Online Backup
Premium Tech Support
Streaming Services
Billing Information
Monthly Charges
Total Charges
Payment Method
Paperless Billing
Contract Type
Target Variable
Churn Label
Yes = 1
No = 0
🔍 Data Cleaning:
Missing Value Handling
Column	Treatment
Churn Reason	Filled with "No Churn"
Churn Category	Filled with "Not Applicable"
Offer	Filled with "No Offer"
Internet Type	Filled with "No Internet Type"
Total Charges	Filled using Median
Total Revenue	Filled using Median
Feature Engineering:
Created:
Age Groups
Tenure Groups
High Usage Customer Flag
Extra Data User Flag


📊 Exploratory Data Analysis (EDA)
Overall Churn Rate
Total Customers: 7,043
Churn Customers: 1,869
Churn Rate: 26.54%

Key Insights:-
📌 Contract Type Impact:
| Contract       | Churn Rate |
| -------------- | ---------- |
| Month-to-Month | 46%        |
| One Year       | 11%        |
| Two Year       | 3%         |
Insight: Customers on month-to-month contracts are significantly more likely to churn.
📌 Internet Type Impact:
| Internet Type | Churn Rate |
| ------------- | ---------- |
| Fiber Optic   | 41%        |
| DSL           | 19%        |
| No Internet   | 7%         |
Insight: Fiber optic users show the highest churn behavior.
📌 Online Security:
| Security | Churn Rate |
| -------- | ---------- |
| No       | 31%        |
| Yes      | 15%        |
Insight: Customers using online security are much less likely to churn.
📌 Premium Tech Support:
| Tech Support | Churn Rate |
| ------------ | ---------- |
| No           | 31%        |
| Yes          | 15%        |
Insight: Tech support strongly improves customer retention.
📌 Paperless Billing:
| Billing Type | Churn Rate |
| ------------ | ---------- |
| Yes          | 34%        |
| No           | 16%        |
Insight: Customers using paperless billing churn at a much higher rate.
📌 Monthly Charges:
| Customer Type | Avg Monthly Charge |
| ------------- | ------------------ |
| Churned       | $74                |
| Retained      | $61                |
Insight: Higher monthly charges are associated with higher churn
📌 Unlimited Data:
| Unlimited Data | Churn Rate |
| -------------- | ---------- |
| Yes            | 32%        |
| No             | 16%        |
Insight: Unlimited data customers churn more frequently.


🤖 Machine Learning Model
Model Used
Random Forest Classifier
Reasons for selection:
Handles categorical and numerical features well
Robust to overfitting
Provides feature importance
Works well on classification problems


Class Imbalance Handling
Used:
SMOTE(random_state=42)
SMOTE generates synthetic samples for the minority class and improves model learning.

ML Pipeline
Pipeline([
    ("processor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier())
])
Steps:
Data Preprocessing
Feature Encoding
Feature Scaling
SMOTE Oversampling
Random Forest Training
Prediction & Evaluation

Model Evaluation Metrics
Evaluated using:
Accuracy Score
F1 Score
Classification Report
Confusion Matrix
ROC-AUC Score
Precision-Recall Curve

📈 Power BI Dashboard
Dashboard KPIs:
Total Customers
Churn Customers
Churn Rate
Churn by Contract
Churn by Internet Type
Churn by Payment Method
Churn by Offers
Churn by Online Security
Churn by Premium Tech Support
Churn by Monthly Charges
Churn by Unlimited Data

💡 Business Recommendations
1. Encourage Long-Term Contracts

Month-to-month customers show the highest churn rate.

2. Promote Online Security

Customers with online security churn almost 50% less.

3. Promote Premium Tech Support

Tech support significantly improves retention.

4. Review Fiber Optic Customer Experience

Fiber optic customers have the highest churn rate.

5. Create Retention Offers

Target high-risk customers with personalized offers.

6. Monitor High Monthly Charges

Customers with higher monthly bills are more likely to leave.    explain how can i explain in data analytics interviews using star
