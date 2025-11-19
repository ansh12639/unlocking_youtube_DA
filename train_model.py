import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('youtube_channel_real_performance_analytics.csv')

# Compute Engagement_Score
df['Engagement_Score'] = (df.get('Likes', 0) + df.get('Shares', 0) + df.get('New Comments', 0)) / df['Views'].replace(0, np.nan) * 100
df['Engagement_Score'] = df['Engagement_Score'].fillna(0)

# Features for revenue prediction (matching app)
features = ['Views', 'Subscribers', 'Engagement_Score']
X = df[features].fillna(0)
y = df['Estimated Revenue (USD)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Revenue model RMSE: {np.sqrt(mse):.4f}, R2: {r2:.4f}')

# Save model
joblib.dump(rf_model, 'revenue_model_rf.pkl')
print('Model saved as revenue_model_rf.pkl')
