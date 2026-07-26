import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.title("Customer Churn Prediction")

# Page Title
st.title("Telecom Customer Churn Analysis")

# Read Dataset
df = pd.read_csv("telecom data.csv")

# Dataset Information
st.header("Dataset Information")

# Shape
st.subheader("Shape of Dataset")
st.write(df.shape)

# Columns
st.subheader("Column Names")
st.write(df.columns.tolist())

# Info
st.subheader("Dataset Info")

import io
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

# Summary Statistics
st.subheader("Summary Statistics")
st.dataframe(df.describe())

# First 5 Rows
st.subheader("First 5 Rows")
st.dataframe(df.head())

import streamlit as st
import pandas as pd

st.header("🧹 Data Cleaning")

# Missing values before cleaning
st.subheader("Missing Values Before Cleaning")
st.dataframe(df.isnull().sum().reset_index().rename(
    columns={"index": "Column", 0: "Missing Values"}
))

# Dataset Info
st.subheader("Dataset Info")

import io
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

# Display churn-related columns
st.subheader("Churn Columns")
st.dataframe(df[["Churn Label", "Churn Category", "Churn Reason"]])

# Fill missing values
df["Churn Reason"] = df["Churn Reason"].fillna("No Churn")
df["Churn Category"] = df["Churn Category"].fillna("Not applicable")
df["Churn Label"] = df["Churn Label"].str.strip()

# Map target column
df["Churn Label"] = df["Churn Label"].map({"Yes": 1, "No": 0})

# Fill numerical missing values
df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())
df["Total Revenue"] = df["Total Revenue"].fillna(df["Total Revenue"].median())

# Fill categorical missing values
df["Offer"] = df["Offer"].fillna("No Offer")
df["Internet Type"] = df["Internet Type"].fillna("No Internet Type")

st.success("✅ Missing values cleaned successfully!")

# Missing values after cleaning
st.subheader("Missing Values After Cleaning")
st.dataframe(df.isnull().sum().reset_index().rename(
    columns={"index": "Column", 0: "Missing Values"}
))

# Preview cleaned columns
st.subheader("Cleaned Churn Columns")
st.dataframe(df[["Churn Label", "Churn Category", "Churn Reason"]].head())

# Display cleaned dataset
st.subheader("Cleaned Dataset")
st.dataframe(df.head(10))

st.header("📊 Exploratory Data Analysis")

st.subheader("Churn Distribution")
count_churnrate = df["Churn Label"].value_counts(normalize=True) * 100
st.write(count_churnrate)
fig, ax = plt.subplots(figsize=(5,4))
sns.countplot(x="Churn Label", data=df, color="gold", ax=ax)
ax.set_title("Churn Distribution")
st.pyplot(fig)

st.subheader("Gender vs Churn")
gender = df.groupby("Gender")["Churn Label"].mean()
st.dataframe(gender)
fig, ax = plt.subplots(figsize=(5,4))
sns.barplot(
    x="Gender",
    y="Churn Label",
    data=df,
    color="skyblue",
    ax=ax
)
st.pyplot(fig)

st.subheader("Age vs Churn")
df["Age"] = pd.cut(
    df["Age"],
    bins=[0,30,60,100],
    labels=["Young","Adult","Senior"]
)
age = df.groupby("Age")["Churn Label"].mean()
st.dataframe(age)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="Age",
    y="Churn Label",
    data=df,
    color="green",
    ax=ax
)
st.pyplot(fig)

st.subheader("📊 Age vs Churn Analysis")

# Age Distribution
st.write("### Age Group Distribution")
st.write(df["Age"].value_counts())

# Average Churn Rate by Age Group
age_by_churn = df.groupby("Age")["Churn Label"].mean()
st.write("### Average Churn Rate by Age Group")
st.dataframe(age_by_churn.reset_index().rename(columns={"Churn Label": "Churn Rate"}))
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="Age",
    y="Churn Label",
    data=df,
    color="pink",
    ax=ax
)
ax.set_xlabel("Age Group")
ax.set_ylabel("Churn Rate")
ax.set_title("Age vs Churn")
st.pyplot(fig)

st.subheader("📊 Under 30 vs Churn Analysis")
under30_by_churn = df.groupby("Under 30")["Churn Label"].mean()
st.write("### Average Churn Rate")
st.dataframe(
    under30_by_churn.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="Under 30",
    y="Churn Label",
    data=df,
    color="pink",
    ax=ax
)
ax.set_xlabel("Under 30")
ax.set_ylabel("Churn Rate")
ax.set_title("Under 30 vs Churn")
st.pyplot(fig)

st.subheader("📊 Senior Citizen vs Churn Analysis")
seniorCitizen_totalRevenue = df.groupby("Senior Citizen")["Churn Label"].mean()
st.write("### Average Churn Rate")
st.dataframe(
    seniorCitizen_totalRevenue.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="Senior Citizen",
    y="Churn Label",
    data=df,
    color="pink",
    ax=ax
)
ax.set_xlabel("Senior Citizen")
ax.set_ylabel("Churn Rate")
ax.set_title("Senior Citizen vs Churn")
st.pyplot(fig)

st.subheader("🏙️ Top 10 Cities by Customer Count")
city_by_churn = df.groupby("City")["Churn Label"].count().sort_values(ascending=False)
top10_cities = city_by_churn.head(10)
st.write("### Top 10 Cities")
st.dataframe(
    top10_cities.reset_index().rename(
        columns={"City": "City", "Churn Label": "Customer Count"}
    )
)
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(
    x=top10_cities.index,
    y=top10_cities.values,
    color="blue",
    ax=ax
)
ax.set_xlabel("City")
ax.set_ylabel("Customer Count")
ax.set_title("Top 10 Cities by Customer Count")
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("🤝 Referred a Friend vs Churn Analysis")
referred_by_churn = df.groupby("Referred a Friend")["Churn Label"].mean()
st.write("### Average Churn Rate")
st.dataframe(
    referred_by_churn.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
st.write("### Churn Summary")
summary = df.groupby("Referred a Friend").agg({
    "Churn Label": ["mean", "count"]
})
st.dataframe(summary)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="Referred a Friend",
    y="Churn Label",
    data=df,
    color="yellow",
    ax=ax
)
ax.set_xlabel("Referred a Friend")
ax.set_ylabel("Churn Rate")
ax.set_title("Referred a Friend vs Churn")
st.pyplot(fig)

st.subheader("📅 Tenure Group vs Churn Analysis")
df["Tenure Group"] = pd.cut(
    df["Tenure in Months"],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=["0-1yr", "1-2yr", "2-3yr", "3-4yr", "4-5yr", "5-6yr"]
)
tenure_by_churn = df.groupby("Tenure Group")["Churn Label"].mean()
st.write("### Average Churn Rate by Tenure Group")
st.dataframe(
    tenure_by_churn.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(
    x="Tenure Group",
    y="Churn Label",
    data=df,
    color="skyblue",
    ax=ax
)
ax.set_xlabel("Tenure Group")
ax.set_ylabel("Churn Rate")
ax.set_title("Churn Rate by Tenure Group")
st.pyplot(fig)

st.subheader("👥 Number of Referrals vs Churn Analysis")
no_of_referrals_by_churn = (
    df.groupby("Number of Referrals")["Churn Label"]
      .mean()
)
st.write("### Average Churn Rate by Number of Referrals")
st.dataframe(
    no_of_referrals_by_churn.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
top5_referrals = no_of_referrals_by_churn.sort_values(
    ascending=False
).head(5)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    x=top5_referrals.index,
    y=top5_referrals.values,
    color="yellow",
    ax=ax
)
ax.set_xlabel("Number of Referrals")
ax.set_ylabel("Churn Rate")
ax.set_title("Top 5 Referral Groups by Churn Rate")
st.pyplot(fig)

st.subheader("🎁 Offer vs Churn Analysis")
offer_by_churn = (
    df.groupby("Offer")["Churn Label"]
      .mean()
      .sort_values(ascending=False)
)
top5_offers = offer_by_churn.head(5)
st.write("### Top 5 Offers by Churn Rate")
st.dataframe(
    top5_offers.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    x=top5_offers.index,
    y=top5_offers.values,
    color="yellow",
    ax=ax
)
ax.set_xlabel("Offer")
ax.set_ylabel("Churn Rate")
ax.set_title("Top 5 Offers vs Churn")
plt.xticks(rotation=20)
st.pyplot(fig)

st.subheader("📞 Phone Service vs Churn Analysis")
phone_services_by_churn = df.groupby("Phone Service")["Churn Label"].mean()
st.write("### Average Churn Rate by Phone Service")
st.dataframe(
    phone_services_by_churn.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="Phone Service",
    y="Churn Label",
    data=df,
    color="pink",
    ax=ax
)
ax.set_xlabel("Phone Service")
ax.set_ylabel("Churn Rate")
ax.set_title("Phone Service vs Churn")
st.pyplot(fig)

st.subheader("🌐 Internet Service vs Churn Analysis")
internet_services = df.groupby("Internet Service")["Churn Label"].mean()
st.write("### Average Churn Rate by Internet Service")
st.dataframe(
    internet_services.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="Internet Service",
    y="Churn Label",
    data=df,
    color="brown",
    ax=ax
)
ax.set_xlabel("Internet Service")
ax.set_ylabel("Churn Rate")
ax.set_title("Internet Service vs Churn")
st.pyplot(fig)

st.subheader("🌐 Internet Type vs Churn Analysis")
internet_type = (
    df.groupby("Internet Type")["Churn Label"]
      .mean()
      .sort_values(ascending=False)
)
st.write("### Average Churn Rate by Internet Type")
st.dataframe(
    internet_type.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
top3_internet = internet_type.head(3)
fig, ax = plt.subplots(figsize=(7,4))
sns.barplot(
    x=top3_internet.index,
    y=top3_internet.values,
    color="green",
    ax=ax
)
ax.set_xlabel("Internet Type")
ax.set_ylabel("Churn Rate")
ax.set_title("Top 3 Internet Types by Churn Rate")
plt.xticks(rotation=15)
st.pyplot(fig)

st.subheader("📶 High Usage vs Churn Analysis")
df["High Usage"] = df["Avg Monthly GB Download"] > 20

high_usage = df.groupby("High Usage")["Churn Label"].mean()
st.write("### Average Churn Rate")
st.dataframe(
    high_usage.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="High Usage",
    y="Churn Label",
    data=df,
    color="yellow",
    ax=ax
)
ax.set_xlabel("High Usage (>20 GB)")
ax.set_ylabel("Churn Rate")
ax.set_title("High Usage vs Churn")
st.pyplot(fig)

st.subheader("🔒 Online Security vs Churn Analysis")
security = df.groupby("Online Security")["Churn Label"].mean()
st.write("### Average Churn Rate")
st.dataframe(
    security.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="Online Security",
    y="Churn Label",
    data=df,
    color="green",
    ax=ax
)
ax.set_xlabel("Online Security")
ax.set_ylabel("Churn Rate")
ax.set_title("Online Security vs Churn")
st.pyplot(fig)
# Business Insight
st.info(
    "💡 Customers without Online Security have a higher churn rate. "
    "Providing Online Security services may help reduce customer churn."
)

st.subheader("💾 Online Backup vs Churn Analysis")
backup = df.groupby("Online Backup")["Churn Label"].mean()
st.write("### Average Churn Rate by Online Backup")
st.dataframe(
    backup.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x="Online Backup",
    y="Churn Label",
    data=df,
    color="orange",
    ax=ax
)
ax.set_xlabel("Online Backup")
ax.set_ylabel("Churn Rate")
ax.set_title("Online Backup vs Churn")
st.pyplot(fig)
# Business Insight
st.info(
    "💡 Customers without Online Backup tend to have a higher churn rate. "
    "Encouraging customers to enable Online Backup may help improve customer retention."
)

st.subheader("🛠️ Premium Tech Support vs Churn Analysis")
tech_support = df.groupby("Premium Tech Support")["Churn Label"].mean()
st.write("### Average Churn Rate by Premium Tech Support")
st.dataframe(
    tech_support.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x="Premium Tech Support",
    y="Churn Label",
    data=df,
    color="pink",
    ax=ax
)
ax.set_xlabel("Premium Tech Support")
ax.set_ylabel("Churn Rate")
ax.set_title("Premium Tech Support vs Churn")
st.pyplot(fig)
# Business Insight
st.success(
    """
**Business Insight**

✅ Customers with **Premium Tech Support** have a much lower churn rate.

- ❌ Without Premium Tech Support → ~31.2% churn
- ✅ With Premium Tech Support → ~15.2% churn

**Recommendation:** Encourage customers to subscribe to Premium Tech Support, as it is associated with significantly better customer retention.
"""
)

st.subheader("📺 Streaming TV vs Churn Analysis")
tv_streaming = df.groupby("Streaming TV")["Churn Label"].mean()
st.write("### Average Churn Rate by Streaming TV")
st.dataframe(
    tv_streaming.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x=tv_streaming.index,
    y=tv_streaming.values,
    color="gold",
    ax=ax
)
ax.set_xlabel("Streaming TV")
ax.set_ylabel("Churn Rate")
ax.set_title("Streaming TV vs Churn")
st.pyplot(fig)
# Business Insight
st.info(
    """
**Business Insight**
📺 Customers who subscribe to **Streaming TV** show a higher churn rate compared to non-users.
**Recommendation:** Review Streaming TV plans, pricing, and bundled offers to improve customer retention.
"""
)

st.subheader("🎬 Streaming Movies vs Churn Analysis")
streaming_movies = df.groupby("Streaming Movies")["Churn Label"].mean()
st.write("### Average Churn Rate by Streaming Movies")
st.dataframe(
    streaming_movies.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x=streaming_movies.index,
    y=streaming_movies.values,
    color="gold",
    ax=ax
)
ax.set_xlabel("Streaming Movies")
ax.set_ylabel("Churn Rate")
ax.set_title("Streaming Movies vs Churn")
st.pyplot(fig)
# Business Insight
st.info(
    """
**Business Insight**

🎬 Customers who subscribe to **Streaming Movies** tend to have a higher churn rate than those who do not.
**Recommendation:** Consider bundling Streaming Movies with loyalty rewards or discounted plans to improve customer retention.
"""
)

st.subheader("🎵 Streaming Music vs Churn Analysis")
streaming_music = df.groupby("Streaming Music")["Churn Label"].mean()
st.write("### Average Churn Rate by Streaming Music")
st.dataframe(
    streaming_music.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x=streaming_music.index,
    y=streaming_music.values,
    color="gold",
    ax=ax
)
ax.set_xlabel("Streaming Music")
ax.set_ylabel("Churn Rate")
ax.set_title("Streaming Music vs Churn")
st.pyplot(fig)
# Business Insight
st.info(
    """
**Business Insight**

🎵 Customers who subscribe to **Streaming Music** tend to have a higher churn rate than non-subscribers.

**Recommendation:** Bundle Streaming Music with discounts, loyalty rewards, or premium service packages to increase customer retention.
"""
)

st.subheader("📶 Unlimited Data vs Churn Analysis")
unlimited_data = df.groupby("Unlimited Data")["Churn Label"].mean()
st.write("### Average Churn Rate by Unlimited Data")
st.dataframe(
    unlimited_data.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x=unlimited_data.index,
    y=unlimited_data.values,
    color="gold",
    ax=ax
)
ax.set_xlabel("Unlimited Data")
ax.set_ylabel("Churn Rate")
ax.set_title("Unlimited Data vs Churn")
st.pyplot(fig)

st.subheader("📄 Contract vs Churn Analysis")
contract = df.groupby("Contract")["Churn Label"].mean()
st.write("### Average Churn Rate by Contract Type")
st.dataframe(
    contract.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(
    x=contract.index,
    y=contract.values,
    color="gold",
    ax=ax
)
ax.set_xlabel("Contract Type")
ax.set_ylabel("Churn Rate")
ax.set_title("Contract Type vs Churn")
plt.xticks(rotation=15)
st.pyplot(fig)

st.subheader("🧾 Paperless Billing vs Churn Analysis")
paperless_billing = df.groupby("Paperless Billing")["Churn Label"].mean()
st.write("### Average Churn Rate by Paperless Billing")
st.dataframe(
    paperless_billing.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x=paperless_billing.index,
    y=paperless_billing.values,
    color="pink",
    ax=ax
)
ax.set_xlabel("Paperless Billing")
ax.set_ylabel("Churn Rate")
ax.set_title("Paperless Billing vs Churn")
st.pyplot(fig)

st.subheader("💳 Payment Method vs Churn Analysis")
payment_method = (
    df.groupby("Payment Method")["Churn Label"]
      .mean()
      .sort_values(ascending=False)
)
st.write("### Average Churn Rate by Payment Method")
st.dataframe(
    payment_method.reset_index().rename(
        columns={"Churn Label": "Churn Rate"}
    )
)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    x=payment_method.index,
    y=payment_method.values,
    color="gold",
    ax=ax
)
ax.set_xlabel("Payment Method")
ax.set_ylabel("Churn Rate")
ax.set_title("Payment Method vs Churn")
plt.xticks(rotation=20)
st.pyplot(fig)


st.subheader("💰 Monthly Charge vs Churn Analysis")
monthly_charge = df.groupby("Churn Label")["Monthly Charge"].mean()
st.write("### Average Monthly Charge")
st.dataframe(
    monthly_charge.reset_index().rename(
        columns={
            "Churn Label": "Churn Status",
            "Monthly Charge": "Average Monthly Charge"
        }
    )
)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x="Churn Label",
    y="Monthly Charge",
    data=df,
    color="skyblue",
    ax=ax
)
ax.set_xlabel("Churn Status (0 = No, 1 = Yes)")
ax.set_ylabel("Average Monthly Charge")
ax.set_title("Monthly Charge vs Churn")
st.pyplot(fig)

st.subheader("📱 Extra Data Usage vs Churn Analysis")
df["Extra Data User"] = df["Total Extra Data Charges"] > 0
extra_data_user = df.groupby("Extra Data User")["Churn Label"].mean()
st.write("### Average Churn Rate by Extra Data Usage")
st.dataframe(
    extra_data_user.reset_index().rename(
        columns={
            "Extra Data User": "Extra Data User",
            "Churn Label": "Churn Rate"
        }
    )
)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x=extra_data_user.index,
    y=extra_data_user.values,
    color="orange",
    ax=ax
)
ax.set_xlabel("Extra Data User")
ax.set_ylabel("Churn Rate")
ax.set_title("Extra Data Usage vs Churn")
st.pyplot(fig)

st.subheader("💰 Total Revenue vs Churn Analysis")
total_revenue = df.groupby("Churn Label")["Total Revenue"].mean()
st.write("### Average Total Revenue by Churn Status")
st.dataframe(
    total_revenue.reset_index().rename(
        columns={
            "Churn Label": "Churn Status",
            "Total Revenue": "Average Total Revenue"
        }
    )
)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x=total_revenue.index,
    y=total_revenue.values,
    color="green",
    ax=ax
)
ax.set_xlabel("Churn Status (0 = No, 1 = Yes)")
ax.set_ylabel("Average Total Revenue")
ax.set_title("Total Revenue vs Churn")
st.pyplot(fig)

st.subheader("📦 Tenure in Months vs Churn")
fig, ax = plt.subplots(figsize=(6,4))
sns.boxplot(
    x="Churn Label",
    y="Tenure in Months",
    data=df,
    ax=ax
)
ax.set_xlabel("Churn Status (0 = No, 1 = Yes)")
ax.set_ylabel("Tenure in Months")
ax.set_title("Customer Tenure vs Churn")
st.pyplot(fig)

st.subheader("🔥 Correlation Heatmap")
corr_matrix = df.corr(numeric_only=True)
st.write("### Correlation Matrix")
st.dataframe(corr_matrix)
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)


# Load model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# -------------------------
# Input Fields
# -------------------------

gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

senior = st.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)

married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure in Months",
    min_value=0,
    max_value=100,
    value=12
)

offer = st.selectbox(
    "Offer",
    [
        "None",
        "Offer A",
        "Offer B",
        "Offer C",
        "Offer D",
        "Offer E"
    ]
)

phone = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["Yes", "No"]
)

internet_type = st.selectbox(
    "Internet Type",
    [
        "DSL",
        "Fiber Optic",
        "Cable",
        "None"
    ]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

device = st.selectbox(
    "Device Protection Plan",
    ["Yes", "No"]
)

tech = st.selectbox(
    "Premium Tech Support",
    ["Yes", "No"]
)

tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly Charge",
    min_value=0.0,
    value=70.0
)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Churn"):

    input_df = pd.DataFrame({

        "Gender":[gender],
        "Age":[age],
        "Senior Citizen":[senior],
        "Married":[married],
        "Dependents":[dependents],
        "Tenure in Months":[tenure],
        "Offer":[offer],
        "Phone Service":[phone],
        "Internet Service":[internet_service],
        "Internet Type":[internet_type],
        "Online Security":[online_security],
        "Online Backup":[online_backup],
        "Device Protection Plan":[device],
        "Premium Tech Support":[tech],
        "Streaming TV":[tv],
        "Streaming Movies":[movies],
        "Contract":[contract],
        "Paperless Billing":[paperless],
        "Payment Method":[payment],
        "Monthly Charge":[monthly]

    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    if probability >= 0.80:
        st.error("High Risk Customer")

    elif probability >= 0.50:
        st.warning("Medium Risk Customer")

    else:
        st.success("Low Risk Customer")