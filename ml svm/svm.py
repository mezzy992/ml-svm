
# SVM classification using Student Performance Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Step 1: Load Dataset
data = pd.read_csv("students.csv")

print("\n===== First 5 Records =====")
print(data.head())

# Step 2: Input Features and Output
X = data[['Study_Hours', 'Attendance', 'Assignments',
          'Internal_Marks', 'Previous_GPA', 'Projects']]

y = data['Result']

# Step 3: Convert Pass/Fail into Numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Step 5: Scale the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train SVM Model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("==============================")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Step 9: Predict New Student
new_student = [[6, 85, 82, 72, 7.8, 2]]

new_student = scaler.transform(new_student)

prediction = model.predict(new_student)

print("\nPrediction for New Student:")
if prediction[0] == 1:
    print("PASS")
else:
    print("FAIL")

# Step 10: Display Confusion Matrix Graph
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=encoder.classes_
)

plt.title("SVM Confusion Matrix")
plt.show()
