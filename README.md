# Talentlock 🔐💼

**Talentlock** is an end-to-end Machine Learning HR Analytics platform designed to predict employee attrition. It helps organizations proactively identify employees at risk of leaving by analyzing various professional metrics, ensuring a cleaner, fairer, and data-driven approach to human resource management.

## 🚀 Features

*   **Machine Learning Prediction Engine:** Utilizes a Random Forest Classification model (outperforming Logistic Regression by over 12%) to predict the likelihood of an employee leaving the company. 
*   **Fairness-First ML:** Strategically removes irrelevant or highly biased demographic data (like `Marital_Status`) to ensure predictions are ethical and based on professional metrics (like `Absenteeism`, `Overtime`, and `Job_Satisfaction`).
*   **Imbalanced Data Handling:** Implements **SMOTE** (Synthetic Minority Oversampling Technique) to mathematically balance the target classes, ensuring realistic, unbiased probability outputs.
*   **Interactive Web App:** A clean, dynamic **Streamlit** UI (light-themed with pastel blue accents). It includes a collapsible **"Advanced Employee Metrics"** expander, allowing HR managers to manipulate up to 20 different variables for highly sensitive on-the-fly predictions. 
*   **Exploratory Data Analysis (EDA):** Deep analytical insights comparing various employee metrics against attrition rates.
*   **PowerBI Integration:** Includes a `.pbix` dashboard for high-level business intelligence reporting and visualization.

---

## 🛠️ Tech Stack
*   **Language:** Python 3.x
*   **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (SMOTE), Matplotlib, Seaborn
*   **Front-End / UI:** Streamlit (`app.py`)
*   **Business Intelligence:** Microsoft PowerBI

---

## 📁 Project Structure

```bash
📦 Talentlock
 ┣ 📜 Talentlock.ipynb             # Phase 1: Data Cleaning & Extensive Exploratory Data Analysis (EDA)
 ┣ 📜 talentlock_after_EDA.ipynb   # Phase 2: Feature Engineering, Model Training, Evaluation, & Exporting
 ┣ 📜 app.py                       # The Streamlit web Application frontend & prediction engine
 ┣ 📜 talentlock_cleaned.csv       # Cleaned and processed dataset used for training
 ┣ 📜 model.pkl                    # Pickled Random Forest prediction model
 ┣ 📜 columns.pkl                  # Pickled columns for the Streamlit UI to match model inputs
 ┗ 📜 TalentLock.pbix              # PowerBI Dashboard visualization file
```

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/talentlock.git
   cd talentlock
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed. You can install the required packages using pip:
   ```bash
   pip install strictly pandas numpy scikit-learn imbalanced-learn matplotlib seaborn streamlit
   ```

3. **Run the Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```
   *This will open the dashboard in your default web browser (typically at `http://localhost:8501`).*

---

## 🧠 Model Performance & Tuning
The primary classification model utilizes a **RandomForestClassifier** dynamically tuned with optimized `max_depth` and `n_estimators`. We chose to prioritize **model fairness and realistic generalizability** over pure brute-force accuracy by discarding non-relevant markers and balancing the data. The resulting model operates at a highly robust real-world accuracy of **~81.8%**.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
