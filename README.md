🧠 ML-Insight Pro: A Toolkit for Advanced Classification Models

This Streamlit application provides an interactive environment to explore and compare six advanced machine learning models for customer churn prediction. It is designed to demonstrate how different classifiers handle class imbalance, visualize model behavior, and interpret performance with real-world datasets.
<img width="1359" height="844" alt="image" src="https://github.com/user-attachments/assets/fbbd55b6-4a60-4037-9a99-17677dcc365a" />
<img width="1344" height="848" alt="image" src="https://github.com/user-attachments/assets/2b7aac8a-2c1e-4437-90b5-404ff04c8cf1" />
<img width="1351" height="854" alt="image" src="https://github.com/user-attachments/assets/afd381bf-8401-4e01-a88b-516fdc1f1ef3" />
<img width="1350" height="941" alt="image" src="https://github.com/user-attachments/assets/392c4f90-e7d0-4968-867b-116f5552ada0" />
<img width="1354" height="862" alt="image" src="https://github.com/user-attachments/assets/d00cdd64-3e39-4c7a-bd9c-5d7933b4183a" />

📌 Features

    Interactive UI built with Streamlit and a custom dark theme.

    Six ML models implemented:

        🌳 Random Forest with hyperparameter tuning.

        ✒️ Support Vector Machine (SVM) with kernel comparison.

        🤖 Neural Network (Keras) with customizable architecture.

        💡 LightGBM with performance tuning.

        🐈 CatBoost for robust classification.

        🚀 XGBoost, a competition-winning classic.

    Flexible Imbalance Handling: Toggle SMOTE (Synthetic Minority Over-sampling Technique) on or off. Models automatically use internal methods (class_weight or scale_pos_weight) when SMOTE is disabled.

    Rich Visualization Tools:

        Confusion matrices for all models.

        Feature importance plots for tree-based models.

        Interactive SVM decision boundary visualization.

        Neural network training/validation curves.

    Easy Navigation: A "Go Back" button on each model page simplifies returning to the home screen.

📂 Dataset

The app uses the Telco Customer Churn dataset, which is a combination of two files:

    churn-bigml-80.csv

    churn-bigml-20.csv

Ensure both files are present in the project's root directory before running the app.

🚀 How to Run

    Clone the repository or download the project files.

    Place the dataset files (churn-bigml-80.csv and churn-bigml-20.csv) in the same directory as the script.

    Install the required libraries. It's recommended to use a requirements.txt file.

    requirements.txt:

    streamlit
    pandas
    numpy
    scikit-learn
    imbalanced-learn
    tensorflow
    matplotlib
    seaborn
    plotly
    lightgbm
    catboost
    xgboost

    Install them using pip:
    Bash

pip install -r requirements.txt

Run the app from your terminal:
Bash

    streamlit run app.py

🧪 Models Overview

Each model page provides a unique set of interactive controls and performance visualizations.

🌳 Random Forest, 💡 LightGBM, 🐈 CatBoost, 🚀 XGBoost

    Tune Hyperparameters: Interactively adjust key parameters like the number of estimators, tree depth, and learning rate.

    Interpret Results: Analyze performance with a detailed classification report, a confusion matrix, and a feature importance plot to see which factors drive churn.

✒️ Support Vector Machine (SVM)

    Compare Kernels: Switch between rbf, linear, and poly kernels to see how they impact classification.

    Visualize Boundaries: An interactive plot shows the decision boundary for any two selected features, offering a clear view of how the SVM separates data points.

🤖 Neural Network (Keras)

    Customize Architecture: Define the number of hidden layers, neurons per layer, and regularization strength.

    Track Training: After training, view plots of training vs. validation accuracy and loss to diagnose overfitting or underfitting.

🛠️ Tech Stack

    Frontend: Streamlit

    ML Frameworks:

        Scikit-learn

        TensorFlow / Keras

        LightGBM

        CatBoost

        XGBoost

        Imbalanced-learn

    Data Visualization:

        Plotly

        Seaborn

        Matplotlib

🧠 What You'll Learn

    How different ML models, from ensembles to neural networks, tackle a classification problem.

    The effect of techniques like SMOTE on handling class imbalance.

    The importance of model interpretability through feature importance plots.

    How to compare the performance and behavior of different gradient boosting implementations (LGBM, CatBoost, XGBoost).

👩‍💻 Author

Vaishnavi Sinha
<br>Made with ❤️.

📜 License

This project is open-sourced under the MIT License.
