# Unlocking YouTube Channel Performance with Data-Driven Intelligence

This project delivers an end-to-end analytics and machine learning pipeline designed to decode the drivers behind YouTube channel growth, revenue generation, and content performance. It integrates data engineering, exploratory dashboards, and a trained Random Forest revenue prediction model to provide actionable insights for creators, analysts, and marketers.

## 🧩 Repository Structure

```
unlocking_youtube_DA/
│
├── YouTube_Optimization_Jupyter_with_Dashboard_Improved_FIXED.ipynb
├── youtube_channel_real_performance_analytics.csv
├── train_model.py
├── revenue_model_rf.pkl
├── requirements_youtube_optimization.txt
└── README.md
```

## ⚙️ Tech Stack

Python 3.13+

Pandas, NumPy

Scikit-Learn

Matplotlib / Seaborn

Streamlit (optional for UI)

Plotly

Joblib

## 🚀 Project Overview

This repository provides a comprehensive workflow for analyzing YouTube performance data using Python and Jupyter Notebooks. It includes:

- A feature-rich dashboard for real-time performance insights  
- A Random Forest machine learning model for revenue prediction  
- Automated feature engineering  
- A complete dataset supporting 70+ YouTube performance metrics  
- A reproducible pipeline powered by the included requirements file  

## 🔍 Key Features

### 1. YouTube Analytics Dashboard (Jupyter Notebook)
The notebook `YouTube_Optimization_Jupyter_with_Dashboard_Improved_FIXED.ipynb` provides:

- Detailed metrics exploration  
- Publish-time optimization (best hour/day to publish)  
- Multi-stream revenue insights  
- Engagement intelligence  
- Auto-generated performance KPIs  

### 2. ML-Based Revenue Prediction Model
`train_model.py` trains a Random Forest model to predict:

- **Estimated Revenue (USD)**

Using features such as:

- Views  
- Subscribers  
- Engagement Score  

Model is saved as `revenue_model_rf.pkl`.

### 3. Complete YouTube Dataset
`youtube_channel_real_performance_analytics.csv` includes:

- Views, Watch Time, Impressions  
- CTR, SEO metrics  
- Revenue breakdown  
- Engagement metrics  
- 70+ performance fields  

## 📥 Installation & Setup

### 1. Clone the repository
```
git clone https://github.com/ansh12639/unlocking_youtube_DA.git
cd unlocking_youtube_DA
```

### 2. Install dependencies
```
pip install -r requirements_youtube_optimization.txt
```

### 3. Open the Jupyter notebook
```
jupyter notebook
```

### 4. Run the ML training script
```
python train_model.py
```

This will retrain and regenerate:

revenue_model_rf.pkl

## 📈 Outputs

- Dashboard insights  
- Predictive revenue model  
- KPI automation  
- Creator strategy insights  

## 📦 Use Cases

- YouTube creators
- Digital marketing analysts
- Revenue modeling
- Data science portfolios
- Creator economy research

## 🤝 Contributing

Contributions, feature requests, and enhancements are welcome.
Feel free to fork the repo and submit pull requests.

## 📜 License

This project is provided for analytical and educational use.
You may adapt models, scripts, and dashboards for your own workflows.
