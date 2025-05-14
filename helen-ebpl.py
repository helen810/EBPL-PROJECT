import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'Parental Education': ['Some College', 'Associate Degree', 'High School', 'Some College', 'Associate Degree'],
    'Lunch': ['Standard', 'Free/Reduced', 'Standard', 'Free/Reduced', 'Standard'],
    'Test Preparation Course': ['Completed', 'None', 'Completed', 'None', 'Completed'],
    'Math Score': [72, 69, 90, 47, 76],
    'Reading Score': [83, 78, 95, 57, 89],
    'Writing Score': [88, 81, 92, 54, 91]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate average score
df['Average Score'] = df[['Math Score', 'Reading Score', 'Writing Score']].mean(axis=1)

# Classify into High, Medium, Low
df['Performance'] = pd.cut(df['Average Score'], bins=[0, 50, 75, 100], labels=['Low', 'Medium', 'High'])

# Encode categorical features
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Parental Education'] = label_encoder.fit_transform(df['Parental Education'])
df['Lunch'] = label_encoder.fit_transform(df['Lunch'])
df['Test Preparation Course'] = label_encoder.fit_transform(df['Test Preparation Course'])
df['Performance'] = label_encoder.fit_transform(df['Performance'])

# Features and target variable
X = df.drop(columns=['Average Score', 'Performance'])
y = df['Performance']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Display predictions
predictions = pd.DataFrame({'Predicted Performance': label_encoder.inverse_transform(y_pred)})
print(predictions)
