import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load Data
data = pd.read_csv('data/house_data.csv')
X = data[['size', 'rooms']]  # Features
y = data['price']           # Target

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save Model
joblib.dump(model, 'model.pkl')
