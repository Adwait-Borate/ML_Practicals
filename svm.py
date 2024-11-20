import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
# Uncomment the following if you want to use cross-validation
# from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("user_behavior_dataset.csv")

# Exploratory Data Analysis
print("First 5 rows of the dataset:\n", df.head())
print("\nDataset statistics:\n", df.describe())
print("\nData types:\n", df.dtypes)
print("\nNull values:\n", df.isnull().sum())

# Data Cleaning
df.drop_duplicates(inplace=True)
df.fillna(df.mean(), inplace=True)

# Data Preprocessing
label_encoder = LabelEncoder()
df['Target'] = label_encoder.fit_transform(df['Target'])  # Encode target variable if necessary

# Separate target variable and features
target_name = 'Target'  # Assuming 'Target' is the name of the target column
target = df[target_name]
data = df.drop(columns=[target_name])

# Feature Scaling
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Implement Support Vector Machine (SVM)
svm = SVC(kernel='linear')  # Using a linear kernel for simplicity
svm.fit(X_train, y_train)

# Making Predictions
svm_pred = svm.predict(X_test)

# Model evaluation
print("Model Training Complete")
print("Train set accuracy:", svm.score(X_train, y_train))
print("Test set accuracy:", svm.score(X_test, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("Classification Report:\n", classification_report(y_test, svm_pred))
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Precision:", precision_score(y_test, svm_pred, average='weighted'))
print("Recall:", recall_score(y_test, svm_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, svm_pred, average='weighted'))