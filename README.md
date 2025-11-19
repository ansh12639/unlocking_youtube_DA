Unlocking YouTube Channel Performance with Data-Driven Intelligence

This project delivers a comprehensive analytics and machine learning workflow designed to decode the drivers behind YouTube channel growth, content performance, and revenue optimization. It integrates data engineering, exploratory dashboards, and a trained Random Forest revenue prediction model to produce actionable outcomes for creators, marketers, and analysts.

🔍 Project Overview

This repository provides an end-to-end framework for analyzing YouTube performance data using Python and Jupyter Notebooks.
It includes:

A feature-rich Jupyter dashboard for real-time insights

A machine learning model (Random Forest) for revenue prediction

A clean dataset engineering layer

A fully reproducible workflow powered by requirements_youtube_optimization.txt

📊 Key Features

1. YouTube Analytics Dashboard (Jupyter Notebook)

The notebook:
YouTube_Optimization_Jupyter_with_Dashboard_Improved_FIXED.ipynb
delivers:

Advanced metrics exploration

Publish time optimization (hour/day-of-week extraction)

Multi-stream revenue evaluation (AdSense, Premium, Playback-based CPM etc.)

Engagement intelligence (likes, comments, shares vs. views)

Auto-generated KPIs and visualizations

Clean dashboards for performance storytelling

2. ML-Based Revenue Forecasting Model

The script:
train_model.py
implements a Random Forest model that predicts:

Estimated Revenue (USD)
from features like:

Views

Subscribers

Engagement Score (auto-calculated)

The output model:
revenue_model_rf.pkl
is ready for deployment and downstream applications.

3. Complete Dataset Included

Dataset used:
youtube_channel_real_performance_analytics.csv

Contains 70+ performance fields:

Views, Watch Time, CTR, Impressions

Ad Revenue breakdown

Playback metrics

User engagement

Content metadata

🧩 Repository Structure
unlocking_youtube_DA/
│
├── YouTube_Optimization_Jupyter_with_Dashboard_Improved_FIXED.ipynb
├── youtube_channel_real_performance_analytics.csv
├── train_model.py
├── revenue_model_rf.pkl
├── requirements_youtube_optimization.txt
└── README.md

⚙️ Tech Stack

Python 3.13+

Pandas, NumPy

Scikit-Learn

Matplotlib / Seaborn

Streamlit (optional for UI)

Plotly

Joblib

📥 Installation & Setup
1. Clone the repository
git clone https://github.com/ansh12639/unlocking_youtube_DA.git
cd unlocking_youtube_DA

2. Install dependencies
pip install -r requirements_youtube_optimization.txt

3. Open the Jupyter notebook
jupyter notebook

4. Run the ML training script
python train_model.py


This will retrain and regenerate:

revenue_model_rf.pkl

📈 Outputs
The project delivers:

High-impact visual analytics

Video performance decomposition

Powerful revenue forecasting

Strategic insights to optimize posting times

A reusable ML model for future YouTube datasets

🚀 Use Cases

Content creators optimizing upload strategy

Digital marketing analytics

Revenue forecasting for YouTube creators

Data science portfolio projects

Creator economy research

🤝 Contributing

Contributions, feature requests, and enhancements are welcome.
Feel free to fork the repo and submit pull requests.

📜 License

This project is provided for analytical and educational use.
You may adapt models, scripts, and dashboards for your own workflows.