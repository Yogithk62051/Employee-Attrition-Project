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
# DATABASE SETUP
# ===============================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            name TEXT,
            age INTEGER,
            marital TEXT,
            income INTEGER,
            gender TEXT,
            prediction TEXT
        )
    ''')

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
# PREDICTION ROUTE
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

    # 2️⃣ Convert dropdown values
    env_enc = int(env)
    job_enc = int(job)
    performance_enc = int(performance)
    worklife_enc = int(worklife)

    # 3️⃣ Encode text values
    marital_enc = encoders['MaritalStatus'].transform([marital])[0]
    gender_enc = encoders['Gender'].transform([gender])[0]

    # 4️⃣ Prepare dataframe (same order as training)
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

    data = pd.DataFrame(
        [[age, marital_enc, income,
          env_enc, gender_enc, job_enc,
          performance_enc, worklife_enc, years]],
        columns=feature_names
    )

    # 5️⃣ Scale data
    data = scaler.transform(data)

    # 6️⃣ Predict
    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Leaving"
        message = "You may be thinking to leave 😟"
    else:
        result = "Retained"
        message = "You are likely to stay in the company 🙂"

    # 7️⃣ Avatar selection
    if gender.lower() == "male":
        avatar = "images/male.png"
    else:
        avatar = "images/female.png"

    # 8️⃣ Save to database
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute(
        "INSERT INTO users VALUES (?,?,?,?,?,?)",
        (username, age, marital, income, gender, result)
    )

    conn.commit()
    conn.close()

    # 9️⃣ Return result to UI
    return render_template(
        "index.html",
        prediction=result,
        message=message,
        avatar=avatar,
        gender=gender
    )


# ===============================
# HISTORY PAGE (ADMIN)
# ===============================
@app.route('/history')
def history():

    # simple admin protection
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