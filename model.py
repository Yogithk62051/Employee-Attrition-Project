import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(r"C:\Users\YOGITH K\Downloads\employee_data.csv")

# ===============================
# ENCODING
# ===============================
label_cols = [
    'Gender',
    'MaritalStatus',
    'EnvironmentSatisfaction',
    'JobSatisfaction',
    'PerformanceRating',
    'WorkLifeBalance'
]

# Encode target column
target_encoder = LabelEncoder()
df['Attrition'] = target_encoder.fit_transform(df['Attrition'])

# Encode categorical features
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# ===============================
# FEATURES & TARGET
# ===============================
X = df[['Age','MaritalStatus','MonthlyIncome',
        'EnvironmentSatisfaction','Gender',
        'JobSatisfaction','PerformanceRating',
        'WorkLifeBalance','YearsAtCompany']]

y = df['Attrition']

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# CLASS WEIGHTS (for imbalance)
# ===============================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ===============================
# MLP NEURAL NETWORK MODEL
# ===============================
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),   # similar to your Dense layers
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# ===============================
# TEST ACCURACY
# ===============================
accuracy = model.score(X_test, y_test)
print("\nModel Test Accuracy:", accuracy)

# Predictions
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# ===============================
# SAVE EVERYTHING
# ===============================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_dict, "encoder.pkl")

print("Model saved successfully!")

print(df['PerformanceRating'].unique())
print(df['JobSatisfaction'].unique())
print(df['EnvironmentSatisfaction'].unique())
print(df['WorkLifeBalance'].unique())