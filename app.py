from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import sqlite3

# Initialize Flask
app = Flask(__name__)

# Load model and preprocessors
model = load_model("model.keras")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoder.pkl")


# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (
                     name
                     TEXT,
                     age
                     INTEGER,
                     marital
                     TEXT,
                     income
                     INTEGER,
                     gender
                     TEXT,
                     prediction
                     TEXT
                 )''')

    conn.commit()
    conn.close()

init_db()


# ---------- HOME PAGE ----------
@app.route('/')
def home():
    return render_template("index.html")


# ---------- PREDICTION ----------
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

    # Convert dropdown values to numbers
    env_enc = int(env)
    job_enc = int(job)
    performance_enc = int(performance)
    worklife_enc = int(worklife)


    # 2️⃣ Encode text → numbers
    marital_enc = encoders['MaritalStatus'].transform([marital])[0]
    gender_enc = encoders['Gender'].transform([gender])[0]

    # 3️⃣ Prepare input
    data = np.array([[age, marital_enc, income,
                      env_enc, gender_enc, job_enc,
                      performance_enc, worklife_enc, years]])

    data = scaler.transform(data)

    # 4️⃣ Predict
    pred = model.predict(data)[0][0]
    result = "Leaving" if pred > 0.35 else "Retained"

    # 5️⃣ Save user input
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users VALUES (?,?,?,?,?,?)",
              (username, age, marital, income, gender, result))
    conn.commit()
    conn.close()

    return render_template("index.html", prediction=result)


# ---------- HISTORY PAGE ----------
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


# ---------- RUN APP ----------
if __name__ == "__main__":
    app.run(debug=True)
