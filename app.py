from flask import Flask, render_template, request
import numpy as np
import joblib
import sqlite3
import pandas as pd
import os

# ===============================
# INITIALIZE FLASK
# ===============================
app = Flask(__name__)

# ===============================
# LOAD MODEL & PREPROCESSORS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))

# ===============================
# DATABASE
# ===============================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (
                     name TEXT,
                     age INTEGER,
                     marital TEXT,
                     income INTEGER,
                     gender TEXT,
                     prediction TEXT
                 )''')

    conn.commit()
    conn.close()

init_db()

# ===============================
# HOME PAGE
# ===============================
@app.route('/')
def home():
    return render_template("index.html")

# ===============================
# PREDICTION
# ===============================
@app.route('/predict', methods=['POST'])
def predict():

    # 1️⃣ Get user input
    username = request.form['username']
    age = int(request.form['age'])
    marital = request.form['marital']
    income = int(request.form['income'])
    env = request.form['env']
    gender = request.form['gender']
    job = request.form['job']
    performance = request.form['performance']
    worklife = request.form['worklife']
    years = int(request.form['years'])

    # Convert dropdown values
    env_enc = int(env)
    job_enc = int(job)
    performance_enc = int(performance)
    worklife_enc = int(worklife)

    # 2️⃣ Encode text
    marital_enc = encoders['MaritalStatus'].transform([marital])[0]
    gender_enc = encoders['Gender'].transform([gender])[0]

    # 3️⃣ Prepare input (DataFrame with feature names)
    feature_names = [
        "Age",
        "MaritalStatus",
        "MonthlyIncome",
        "EnvironmentSatisfaction",
        "Gender",
        "JobSatisfaction",
        "PerformanceRating",
        "WorkLifeBalance",
        "YearsAtCompany"
    ]

    data = pd.DataFrame([[age, marital_enc, income,
                          env_enc, gender_enc, job_enc,
                          performance_enc, worklife_enc, years]],
                        columns=feature_names)

    data = scaler.transform(data)

    # 4️⃣ Predict (sklearn MLP)
    pred = model.predict(data)[0]
    result = "Leaving" if pred == 1 else "Retained"

    # 5️⃣ Save to database
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users VALUES (?,?,?,?,?,?)",
              (username, age, marital, income, gender, result))
    conn.commit()
    conn.close()

    return render_template("index.html", prediction=result)

# ===============================
# HISTORY PAGE
# ===============================
@app.route('/history')
def history():
    if request.args.get("key") != "yogithk62051":
        return "Access Denied"

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users")
    data = c.fetchall()
    conn.close()

    return render_template("history.html", data=data)

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)