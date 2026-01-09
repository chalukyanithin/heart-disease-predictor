# ❤️ Heart Disease Risk Predictor

An AI-powered web application that predicts the risk of heart disease using clinical patient data.
Built with **Machine Learning + Streamlit**, this system provides both **manual input** and **CSV-based bulk prediction** in a modern, user-friendly interface.

---

## 🧠 What this project does

This application allows users (doctors, students, researchers) to:

* Enter patient health data manually
* Get instant predictions of heart disease risk
* View the estimated probability of having heart disease

The system uses a trained Machine Learning model built on the **Kaggle Heart Disease dataset**.

---

## 🚀 Live Features

* ✅ AI-based heart disease prediction
* 🧍 Manual patient data entry
* 📊 Risk percentage estimation
* 🌙 Modern dark UI
* 🧪 Built for educational & clinical demo use

---

## 🖥️ Screenshots

### 🔹 Home Screen

![Home](Screenshot%20%28226%29.png)

---

### 🔹 Prediction Result

![Result](Screenshot%20%28228%29.png)

---

## 🛠 Tech Stack

| Layer      | Technology                          |
| ---------- | ----------------------------------- |
| Frontend   | Streamlit                           |
| Backend    | Python                              |
| ML Model   | Logistic Regression / Random Forest |
| Data       | Kaggle Heart Disease Dataset        |
| UI         | Custom dark theme                   |
| Deployment | Streamlit Cloud / Localhost         |

---

## 📂 Project Structure

```
heart-disease-predictor/
│
├── app.py                # Main Streamlit app
├── model.pkl             # Trained ML model
├── scaler.pkl            # Feature scaler
├── requirements.txt     # Python dependencies
├── dataset.csv           # Training dataset
└── README.md
```

---

## 🧪 Input Features Used

The model predicts risk based on:

* Age
* Sex
* Chest pain type
* Blood pressure
* Cholesterol
* Fasting blood sugar
* ECG results
* Maximum heart rate
* Exercise-induced angina
* ST depression
* Thalassemia
* Number of major vessels

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/chalukyanithin/heart-disease-predictor
cd heart-disease-predictor
pip install -r requirements.txt
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 📊 Output

The system gives:

* **Low Risk** or **High Risk**
* A **percentage probability** (e.g., 45%)

> ⚠️ This is an AI-assisted prediction system and **not a medical diagnosis**.

---

## 🎯 Use Cases

* Medical students
* Data science projects
* Clinical risk analysis
* Machine learning demos
* Healthcare AI portfolios

---

## 🧠 Dataset Source

Kaggle Heart Disease Dataset
Used only for **educational and research purposes**

---

## 🧑‍💻 Author

**Nithyananda (Chalukya Nithin)**
Computer Science Engineer | AI & Systems Enthusiast

GitHub: [https://github.com/chalukyanithin](https://github.com/chalukyanithin)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!

---

If you want, I can also help you add:

* Model accuracy section
* ROC curve
* Or a professional portfolio description 🔥
