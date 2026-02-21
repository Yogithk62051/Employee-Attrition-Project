import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

df = pd.read_csv(r"C:\Users\YOGITH K\Downloads\employee_data.csv")

# ML models understand numbers, not text. so converting these two columns into binary
label_cols = [
    'Gender',
    'MaritalStatus',
    'EnvironmentSatisfaction',
    'JobSatisfaction',
    'PerformanceRating',
    'WorkLifeBalance'
]
# Encode target column (Attrition)
target_encoder = LabelEncoder()
df['Attrition'] = target_encoder.fit_transform(df['Attrition'])

le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# input and output X = features , y = target
X = df[['Age','MaritalStatus','MonthlyIncome',
        'EnvironmentSatisfaction','Gender',
        'JobSatisfaction','PerformanceRating',
        'WorkLifeBalance','YearsAtCompany']]

y = df['Attrition']

# Scale Data MLP works better when data is normalized.
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build MLP Model , this line Start's neural network.
model = Sequential()

# Hidden Layers
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))

#Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)


class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    class_weight=class_weights
)

# ---------- TEST ACCURACY ----------
loss, accuracy = model.evaluate(X_test, y_test)

print("\nModel Test Accuracy:", accuracy)

# Evaluation
from sklearn.metrics import classification_report

y_pred = (model.predict(X_test) > 0.5).astype(int)

print(classification_report(y_test, y_pred))

# Save Model
model.save("model.keras")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_dict, "encoder.pkl")

print(df['PerformanceRating'].unique())
print(df['JobSatisfaction'].unique())
print(df['EnvironmentSatisfaction'].unique())
print(df['WorkLifeBalance'].unique())

