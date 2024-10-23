import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Setting seed for reproducibility
np.random.seed(42)

# Generating random data
person_id = range(1, 21)  # IDs for 20 people
no_of_km = np.round(np.random.uniform(1, 20, size=20), 1)  # Kilometers run between 1 and 20
time_taken = np.round(no_of_km * np.random.uniform(5, 10, size=20), 2)  # Time taken (based on distance)
wt = np.round(np.random.uniform(50, 90, size=20), 1)  # Weight of the person between 50 and 90 kg

# Creating the dataframe
df = pd.DataFrame({
    'person_id': person_id,
    'no_of_km': no_of_km,
    'time_taken': time_taken,
    'wt': wt
})

# Display the dataframe
print(df)
print("------------------------------------------------------------------------")

# dropping unnessecary data
df = df.drop(columns = ["person_id"])

# separating target and predictors
X = df.drop(columns = ["wt"])
y = df["wt"]

# test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model building

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using X_train and y_train
model.fit(X_train, y_train)

# Predict the target values for X_test
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared

# Print evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print("------------------------------------------------------------------------")

# pickling the model
with open ("model.pkl", "wb") as file:
    pickle.dump(model, file)
    

# creating a database to pickle the testing data    
db = {'X_test' : X_test,
      'y_test' : y_test
      }

# pickling the testing data
with open ("data.pkl", "wb") as file:
    pickle.dump(db, file)




