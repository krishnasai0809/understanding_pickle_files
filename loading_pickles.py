import pickle
from sklearn.metrics import mean_squared_error, r2_score

# unpickling the testing data
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    
X_test = loaded_data['X_test']
y_test = loaded_data['y_test']

# loading the model back
with open ("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Predict the target values for X_test
y_pred = loaded_model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared

# Print evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


