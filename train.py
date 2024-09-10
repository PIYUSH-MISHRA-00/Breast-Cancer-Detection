import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load and clean data
data = pd.read_csv('data.csv')
data = data.drop(columns=['Unnamed: 32'], errors='ignore')  # Drop unwanted column
X = data.drop(['diagnosis'], axis=1)  # Assuming 'diagnosis' is the target
y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Malignant: 1, Benign: 0

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the model to a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the columns used for training
with open('model_columns.pkl', 'wb') as file:
    pickle.dump(X_train.columns, file)
