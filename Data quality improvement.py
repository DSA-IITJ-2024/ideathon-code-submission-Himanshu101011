import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess datasets

# Load drug interaction dataset
drug_interaction_data = pd.read_csv('drug_interaction_data.csv')

# Load adverse effect dataset
adverse_effect_data = pd.read_csv('adverse_effect_data.csv')

# Load real-world evidence dataset
real_world_evidence_data = pd.read_csv('real_world_evidence_data.csv')

# Preprocess datasets (handle missing values, standardize data formats, etc.)
drug_interaction_data.dropna(inplace=True)
adverse_effect_data.dropna(inplace=True)
real_world_evidence_data.dropna(inplace=True)

# Merge datasets based on common columns (e.g., drug name)
merged_data = pd.merge(drug_interaction_data, adverse_effect_data, on='Drug', how='inner')

# Step 2: Feature Engineering and Model Training

# Extract features from merged dataset
X = merged_data[['Drug', 'InteractingDrug', 'Effect']]
y = merged_data['InteractionType']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature engineering (e.g., encoding categorical variables, transforming data)
# For simplicity, let's assume encoding is not needed in this example

# Initialize and train a machine learning model (e.g., Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Model Evaluation

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 4: Incorporating Real-World Evidence

# Merge real-world evidence data with existing dataset
merged_data_with_rwe = pd.merge(merged_data, real_world_evidence_data, on='PatientID', how='inner')

# Further feature engineering incorporating real-world evidence (e.g., patient characteristics)
X_with_rwe = merged_data_with_rwe[['Drug', 'InteractingDrug', 'Effect', 'Age', 'Gender']]
y_with_rwe = merged_data_with_rwe['InteractionType']

# Re-train the model with additional features from real-world evidence
X_train_rwe, X_test_rwe, y_train_rwe, y_test_rwe = train_test_split(X_with_rwe, y_with_rwe, test_size=0.2, random_state=42)
model_with_rwe = RandomForestClassifier(n_estimators=100, random_state=42)
model_with_rwe.fit(X_train_rwe, y_train_rwe)

# Make predictions on the testing set using the model with real-world evidence
y_pred_rwe = model_with_rwe.predict(X_test_rwe)

# Evaluate the model's performance with real-world evidence
accuracy_rwe = accuracy_score(y_test_rwe, y_pred_rwe)
print("Accuracy with real-world evidence:", accuracy_rwe)
