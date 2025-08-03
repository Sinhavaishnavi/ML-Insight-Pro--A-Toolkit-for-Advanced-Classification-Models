

---

# 🧠Churn Prediction Web App

This Streamlit application provides an interactive environment to explore and compare **three advanced machine learning models** for **customer churn prediction**. It is designed to demonstrate how different classifiers handle **class imbalance**, visualize decision boundaries, and interpret model performance with real-world datasets.

---

## 📌 Features

* **Interactive UI built with Streamlit**
* **Three ML models** implemented:

  * 🌳 **Random Forest** with hyperparameter tuning and feature importance
  * ✒️ **Support Vector Machine (SVM)** with kernel comparison and decision boundary visualization
  * 🤖 **Neural Network (Keras)** with customizable architecture and training metrics
* **Imbalanced dataset handling** using `class_weight='balanced'`
* **Visualization tools**:

  * Confusion matrix
  * Feature importance bar plot
  * SVM decision boundaries
  * Neural network training loss and accuracy over epochs
* Custom **dark theme UI** for a polished look

---

## 📂 Dataset

The app uses the **Telco Customer Churn** dataset from Kaggle, split into:

* `churn-bigml-80.csv` (80% training)
* `churn-bigml-20.csv` (20% testing)

Ensure both files are placed in the project directory.

---

## 🚀 How to Run

1. **Clone the repository** or download the files.
2. Place `churn-bigml-80.csv` and `churn-bigml-20.csv` in the root directory.
3. Install the required libraries:

   
4. Run the app:

   ```bash
   streamlit run your_script_name.py
   ```

---

## 🧪 Model Performance Summary

### 🌳 Random Forest

* **Hyperparameters**: `n_estimators=150`, `max_depth=10`
* **Handling Imbalance**: `class_weight='balanced'`
* **Performance**:

  * Precision, Recall, F1-score shown per class
  * 5-Fold CV Accuracy (e.g.): `~84% ± 1.2%`

### ✒️ Support Vector Machine (SVM)

* **Kernel**: RBF / Linear / Polynomial (user-selectable)
* **Performance (on all features)**:

  * Accuracy: **76.38%**
  * Precision: **36.23%**
  * Recall: **82.64%**
  * F1-Score: **50.38%**
  * AUC Score: **0.847**
* **Interactive boundary plot** for any 2 features

### 🤖 Neural Network (Keras)

* **Architecture**: Configurable layers, dropout
* **Class Weighting**: Automatically applied
* **Test Accuracy**: **87.77%**
* **Additional Outputs**:

  * Training vs Validation loss
  * Training vs Validation accuracy
  * Classification Report

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **ML Frameworks**:

  * Scikit-learn
  * TensorFlow / Keras
* **Data Visualization**:

  * Plotly
  * Seaborn
  * Matplotlib
* **Styling**: Custom dark theme via embedded CSS

---

## 🧠 What You'll Learn

* How different ML models behave under class imbalance
* Importance of model interpretability (e.g., feature importances)
* Visual decision boundaries in high-dimensional classification
* How neural networks can outperform traditional models in certain use-cases

---

## 👩‍💻 Author

**Vaishnavi Sinha**
Made with ❤️ .

---

## 📜 License

This project is open-sourced under the MIT License.

---


