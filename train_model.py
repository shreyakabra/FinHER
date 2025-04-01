import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# Dummy Training Data
X_train = np.random.rand(100, 4)
y_train = np.random.randint(300, 850, 100)

# Train a simple credit scoring model
model = RandomForestRegressor(n_estimators=10)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "credit_model.pkl")
print("Model trained and saved successfully!")
