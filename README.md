# 🚕 NYC Taxi Fare Analytics & Prediction Pipeline

End-to-end big data analytics and machine learning pipeline built on the NYC TLC Taxi dataset using Databricks Spark, SQL, and Scikit-learn.

The project processes and analyzes over **958 million taxi trips** to uncover mobility patterns, business insights, and predict taxi fares at scale using machine learning models.

---

## 📊 Dashboard Preview

![NYC Taxi Dashboard](./assets/NYC_taxt__analytics_dashboard)

---

## 📌 Project Overview

This project demonstrates a complete big data engineering workflow:

- Large-scale data ingestion and cleaning with Spark
- SQL analytics for business insights
- Feature engineering and preprocessing
- Machine learning fare prediction
- Model evaluation using RMSE

The final cleaned dataset contains:

- **958.3M+ total taxi trips**
- **879.4M yellow taxi trips**
- **78.9M green taxi trips**

---

## 🛠️ Tech Stack

- Python
- Databricks
- Apache Spark
- SQL
- Pandas
- Scikit-learn
- XGBoost
- Parquet

---

## 🧹 Data Cleaning & Processing

The pipeline removes unrealistic and invalid trips, including:

- Negative trip durations
- Invalid pickup/dropoff timestamps
- Unrealistic speeds
- Extreme trip distances
- Invalid passenger counts
- Non-positive fares
- Duplicate records

After cleaning:

- Green Taxi Valid Trips: **78.9M**
- Yellow Taxi Valid Trips: **879.4M**

---

## 📈 Business Analytics

Key business analyses include:

- Monthly trip trends
- Peak travel hours
- Revenue analysis
- Trip duration distribution
- Tip behavior analysis
- Borough-to-borough revenue flows
- Driver earning efficiency

### 🔍 Key Insights

- Manhattan → Manhattan trips generated over **61%** of total revenue
- Nearly **63%** of trips included tips
- Most trips were completed within **5–20 minutes**
- 5–10 minute trips provided the best balance between demand and earning efficiency

---

## 🤖 Machine Learning Models

The project predicts taxi `total_amount` using trip features such as:

- Trip distance
- Passenger count
- Pickup/dropoff locations
- Time-based features
- Taxi type
- Payment type
- RateCodeID

### 📉 Models Used

| Model | Test RMSE |
|---|---|
| Baseline Model | 10.31 |
| ElasticNet | 6.76 |
| XGBoost Regressor | **5.40** |

### 🏆 Best Model

XGBoost achieved the best performance by capturing complex non-linear relationships in large-scale trip data.

---

## 📂 Repository Structure

```bash
├── assets/
│   ├── NYC_taxi_analytics_dashboard.png
│   └── placeholder.txt
│
├── datasets/
│   ├── Download Dataset.ipynb
│   ├── taxi_zone_lookup.csv
│   └── placeholder.txt
│
├── notebooks/
│   ├── Data_Ingestion_and_Preparation.ipynb
│   ├── Business_insights.py
│   ├── ML_models_for_prediction(0).py
│   ├── ML_models_for_prediction(1).py
│   ├── ML_models_for_prediction(2).py
│   ├── ML_models_for_prediction(3).py
│   └── placeholder.txt
│
├── report/
│   ├── Project_report.docx
│   └── placeholder.txt
│
├── .gitignore
├── LICENSE
├── Project Brief.docx
└── README.md
```

---
## 🚀 Future Improvements
- Use Spark MLlib for distributed model training
- Add weather and traffic datasets
- Deploy real-time prediction pipeline
- Build interactive dashboard with Streamlit or Power BI
- Automate hyperparameter tuning

---
## 👤 Author

**Ngoc Bao Tran Nguyen**
Master of Data Science and Innovation
