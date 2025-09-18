# 📌 E-commerce Purchase Prediction using Logistic Regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =======================
# 1. Load Dataset
# =======================
df = pd.read_csv("ecommerce_purchase_dataset.csv")

# =======================
# 2. Preprocessing
# =======================
# Encode categorical variables
le_device = LabelEncoder()
df["Device_Type"] = le_device.fit_transform(df["Device_Type"])

le_region = LabelEncoder()
df["Region"] = le_region.fit_transform(df["Region"])

# Define features (X) and target (y)
X = df.drop(["CustomerID", "Purchase"], axis=1)
y = df["Purchase"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =======================
# 3. Model Training
# =======================
model = LogisticRegression()
model.fit(X_train, y_train)

# =======================
# 4. Predictions
# =======================
y_pred = model.predict(X_test)

# =======================
# 5. Evaluation
# =======================
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["No Purchase", "Purchase"], yticklabels=["No Purchase", "Purchase"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
