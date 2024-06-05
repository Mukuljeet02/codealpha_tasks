# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Load the dataset
titanic_data = pd.read_csv('titanic.csv')

# Step 3: Data Preprocessing
# Handle missing values
titanic_data.dropna(subset=['Age'], inplace=True)
titanic_data.fillna({'Cabin': 'Unknown', 'Embarked': 'Unknown'}, inplace=True)

# Convert categorical variables into numerical
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Step 4: Feature Selection
features = ['Pclass', 'Sex', 'Age']

# Step 5: Split Data
X = titanic_data[features]
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Choose a Model
model = LogisticRegression()

# Step 7: Train the Model
model.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 9: Make Predictions
# Example prediction
example_passenger = [[3, 0, 25]]  # Example passenger: Pclass 3, Male, Age 25
prediction = model.predict(example_passenger)
print("Prediction:", prediction)
