# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Import New Libraries ---
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- App Configuration and Styling ---
st.set_page_config(
    page_title="ML-Insight Pro: A Toolkit for Advanced Classification Models",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for a Polished Dark Theme
def inject_custom_css():
    css = """
    <style>
        body { color: #f1f1f1; background-color: #111111; }
        .main { background-color: #111111 !important; }
        [data-testid="stSidebar"] { background-color: #222222 !important; border-right: 2px solid #333333; }
        [data-testid="stSidebar"] * { color: #f1f1f1 !important; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #8055BD !important; }
        .stButton>button {
            border: 2px solid #8055BD !important; background-color: #8055BD !important; color: white !important;
            border-radius: 12px !important; font-weight: 700 !important; transition: all 0.3s ease !important;
            box-shadow: 0 4px 8px rgba(149, 81, 184, 0.2);
        }
        .stButton>button:hover {
            background-color: #222222 !important; color: #8055BD !important;
            border-color: #8055BD !important; box-shadow: 0 0 10px #8055BD !important;
        }
        .stMetric { background-color: #222222 !important; border: 1.5px solid #333333; border-radius: 14px; }
        h1, h2, h3 { color: #8055BD !important; font-weight: 700; }
        .st-expander, .st-expander header { background-color: #222222 !important; color: #f1f1f1 !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Caching Functions for Performance ---
@st.cache_data
def load_churn_data():
    """Loads, preprocesses, and engineers features for the customer churn dataset."""
    try:
        df_80 = pd.read_csv('churn-bigml-80.csv')
        df_20 = pd.read_csv('churn-bigml-20.csv')
        df = pd.concat([df_80, df_20], ignore_index=True)
        df.drop_duplicates(inplace=True)

        # Basic Preprocessing
        le = LabelEncoder()
        df['State'] = le.fit_transform(df['State'])
        df['International plan'] = df['International plan'].map({'No': 0, 'Yes': 1})
        df['Voice mail plan'] = df['Voice mail plan'].map({'No': 0, 'Yes': 1})
        df['Churn'] = df['Churn'].astype(int)

        # --- Feature Engineering ---
        # Avoid division by zero by replacing 0 with a small number
        df['Total day minutes'] = df['Total day minutes'].replace(0, 0.01)
        df['Account length'] = df['Account length'].replace(0, 1)

        df['cost_per_day_minute'] = df['Total day charge'] / df['Total day minutes']
        df['calls_per_account_day'] = df['Customer service calls'] / df['Account length']
        df['intl_usage_ratio'] = df['Total intl calls'] / (df['Total intl minutes'] + 0.01)
        
        # Fill any potential NaN/inf values created
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        features = df.drop('Churn', axis=1).columns.tolist()
        target = 'Churn'
        
        X = df[features]
        y = df[target]
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

        y_labels = ['No Churn', 'Churn']
        
        return df, train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y), features, y_labels
    except FileNotFoundError:
        st.error("Error: Make sure 'churn-bigml-80.csv' and 'churn-bigml-20.csv' are in the correct directory.")
        return None, (None, None, None, None), None, None

# --- Main Application Logic ---
def main():
    inject_custom_css()
    
    st.sidebar.title("🧠 ML-Insight Pro")
    st.sidebar.markdown("---")

    # Define pages
    PAGES = [
        "🏠 Home Page",
        "🌳 Random Forest Classifier",
        "✒️ Support Vector Machine",
        "🤖 Neural Network (Keras)",
        "💡 LightGBM Classifier",
        "🐈 CatBoost Classifier",
        "🚀 XGBoost Classifier"
    ]
    
    # Use a key for the radio widget to allow programmatic changes
    app_mode = st.sidebar.radio("Select a Page", PAGES, key="page_selector")
    
    st.sidebar.markdown("---")
    
    # Global settings for all models
    st.sidebar.header("Global Model Settings")
    use_smote = st.sidebar.checkbox("Use SMOTE for Imbalance?", value=False, help="Synthetically creates new data points for the minority 'Churn' class to balance the dataset.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ by Vaishnavi Sinha")
    st.sidebar.markdown("Dataset: Telco Churn from Kaggle")

    df, (X_train, X_test, y_train, y_test), features, y_labels = load_churn_data()
    
    if df is None:
        return

    # Apply SMOTE if selected
    if use_smote:
        with st.spinner("Applying SMOTE to balance training data..."):
            smote = SMOTE(random_state=42)
            X_train_processed, y_train_processed = smote.fit_resample(X_train, y_train)
    else:
        X_train_processed, y_train_processed = X_train, y_train


    if app_mode == "🏠 Home Page":
        page_introduction(df)
    elif app_mode == "🌳 Random Forest Classifier":
        page_random_forest(X_train_processed, X_test, y_train_processed, y_test, features, y_labels, use_smote)
    elif app_mode == "✒️ Support Vector Machine":
        page_svm(X_train_processed, X_test, y_train_processed, y_test, features, y_labels, use_smote)
    elif app_mode == "🤖 Neural Network (Keras)":
        page_neural_network(X_train_processed, X_test, y_train_processed, y_test, features, y_labels, use_smote)
    elif app_mode == "💡 LightGBM Classifier":
        page_lightgbm(X_train_processed, X_test, y_train_processed, y_test, features, y_labels, use_smote)
    elif app_mode == "🐈 CatBoost Classifier":
        page_catboost(X_train_processed, X_test, y_train_processed, y_test, features, y_labels, use_smote)
    elif app_mode == "🚀 XGBoost Classifier":
        page_xgboost(X_train_processed, X_test, y_train_processed, y_test, features, y_labels, use_smote)


def page_introduction(df):
    st.title("🏠 Welcome to ML-Insight Pro!")
    st.markdown("This application provides an interactive way to explore and compare several advanced machine learning models for predicting customer churn.")
    
    st.markdown("### What Each Model Does:")
    st.info("""
    - **🌳 Random Forest Classifier:** An ensemble model that builds multiple decision trees and merges them to get a more accurate and stable prediction.
    - **✒️ Support Vector Machine (SVM):** A powerful classifier that finds the optimal hyperplane to separate data points into different classes.
    - **🤖 Neural Network (Keras):** A simple deep learning model that learns complex patterns in the data through interconnected layers of 'neurons'.
    - **💡 LightGBM Classifier:** A fast, high-performance gradient boosting framework that uses tree-based learning algorithms.
    - **🐈 CatBoost Classifier:** A gradient boosting library that is particularly effective at handling categorical features automatically.
    - **🚀 XGBoost Classifier:** An optimized and popular gradient boosting library, known for its performance and accuracy in competitions.
    """)
    
    st.markdown("### ⚠️ The Challenge of Class Imbalance")
    st.warning("""
    In this dataset, there are far more 'No Churn' customers than 'Churn' customers. This is called **class imbalance**. 
    A naive model might achieve high accuracy by simply always predicting 'No Churn', but it would be useless for identifying customers who are actually at risk. 
    To fix this, all models in this app use techniques like **class weights**, **scale_pos_weight**, or **SMOTE** (Synthetic Minority Over-sampling Technique) to address this issue.
    """)
    
    st.markdown("### 📊 Dataset Preview")
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("**Dataset Sample**")
        st.dataframe(df.head())
    with col2:
        st.markdown("**Churn Distribution**")
        churn_counts = df['Churn'].value_counts().rename({0: 'No Churn', 1: 'Churn'})
        st.dataframe(churn_counts)
        fig = px.pie(values=churn_counts.values, names=churn_counts.index, title="Churn vs. No Churn", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show Dataset Summary Statistics"):
        st.dataframe(df.describe())

# ---
# Helper function for "Go Back" button
# ---
def go_back_button():
    """Creates a button that, when clicked, navigates back to the home page."""
    def go_home():
        st.session_state.page_selector = "🏠 Home Page"
    st.button("Go Back to Home Page", on_click=go_home)

# ---
# Random Forest Page
# ---
def page_random_forest(X_train, X_test, y_train, y_test, features, y_labels, use_smote):
    st.title("🌳 Random Forest for Churn Prediction")
    go_back_button() # Add the go back button
    st.markdown(f"Training a Random Forest to predict whether a customer will **churn**. This model is using {'SMOTE' if use_smote else '`class_weight=balanced`'} to handle imbalanced data.")
    
    st.sidebar.header("Hyperparameter Tuning")
    n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 150, 50, help="The number of decision trees in the forest.")
    max_depth = st.sidebar.slider("Max Depth of Trees", 2, 20, 10, 1, help="The maximum depth of each decision tree.")
    
    class_weight_param = None if use_smote else 'balanced'
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1, class_weight=class_weight_param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.subheader("📊 Model Performance")
    report = classification_report(y_test, y_pred, target_names=y_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Feature Importance")
        importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=True)
        fig = px.bar(importances.tail(15), x='importance', y='feature', orientation='h', title="Top 15 Feature Importances", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🌀 Confusion Matrix")
        with plt.style.context('dark_background'):
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=y_labels, yticklabels=y_labels, ax=ax)
            ax.set_title('Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            st.pyplot(fig, clear_figure=True)

# ---
# SVM Page
# ---
@st.cache_resource(show_spinner="Training main SVM model...")
def train_svm_model(X, y, kernel, C, gamma, class_weight_param):
    """Trains and caches the main SVM model."""
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42, class_weight=class_weight_param)
    model.fit(X, y)
    return model

@st.cache_data(show_spinner="Generating decision boundary plot...")
def get_decision_boundary_figure(X_train, y_train, feature_x, feature_y, kernel, C, gamma, class_weight_param):
    """Trains a 2D SVM for visualization and generates the plot. Caches the resulting figure."""
    X_train_vis = X_train[[feature_x, feature_y]]
    model_vis = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42, class_weight=class_weight_param)
    model_vis.fit(X_train_vis, y_train)
    x_min, x_max = X_train_vis[feature_x].min() - 1, X_train_vis[feature_x].max() + 1
    y_min, y_max = X_train_vis[feature_y].min() - 1, X_train_vis[feature_y].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
        ax.scatter(X_train_vis[feature_x], X_train_vis[feature_y], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k', alpha=0.7)
        ax.set_xlabel(feature_x); ax.set_ylabel(feature_y); ax.set_title('SVM Decision Boundary')
    return fig

def page_svm(X_train, X_test, y_train, y_test, features, y_labels, use_smote):
    st.title("✒️ Support Vector Machine for Churn Prediction")
    go_back_button()
    st.markdown(f"Training an SVM for churn prediction. This model is using {'SMOTE' if use_smote else '`class_weight=balanced`'} to handle imbalanced data.")
    
    st.sidebar.header("Hyperparameter Tuning")
    kernel = st.sidebar.selectbox("Kernel", ['rbf', 'linear', 'poly'], index=0)
    C = st.sidebar.select_slider("Regularization (C)", options=[0.1, 1, 10, 100], value=1)
    gamma = st.sidebar.select_slider("Gamma", options=['scale', 'auto', 0.01, 0.1, 1], value='scale')
    
    st.sidebar.header("Decision Boundary Visualization")
    feature_x = st.sidebar.selectbox("X-axis Feature", features, index=features.index('Total day minutes'))
    feature_y = st.sidebar.selectbox("Y-axis Feature", features, index=features.index('Customer service calls'))

    class_weight_param = None if use_smote else 'balanced'
    model_full = train_svm_model(X_train, y_train, kernel, C, gamma, class_weight_param)
    y_pred = model_full.predict(X_test)
    y_proba = model_full.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📈 Decision Boundary")
        fig = get_decision_boundary_figure(X_train, y_train, feature_x, feature_y, kernel, C, gamma, class_weight_param)
        st.pyplot(fig, clear_figure=True)
    with col2:
        st.subheader("📊 Model Performance")
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        st.metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")
        st.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.3f}")
        st.markdown("**Classification Report**")
        st.text(classification_report(y_test, y_pred, target_names=y_labels))

# ---
# Neural Network Page
# ---
def page_neural_network(X_train, X_test, y_train, y_test, features, y_labels, use_smote):
    st.title("🤖 Neural Network for Churn Prediction")
    go_back_button()
    st.markdown("Training a simple neural network. This model uses advanced callbacks and regularization to improve performance.")

    st.sidebar.header("Network Architecture")
    hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)
    neurons = st.sidebar.slider("Neurons per Hidden Layer", 16, 128, 64, 16)
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 15)
    l2_reg = st.sidebar.select_slider("L2 Regularization", options=[0.0, 0.001, 0.01, 0.1], value=0.01)

    model = Sequential([
        Dense(neurons, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(l2_reg))
    ])
    for i in range(hidden_layers - 1):
        model.add(Dense(max(8, neurons // (i+2)), activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    class_weights = None
    if not use_smote:
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(weights))
    
    if st.button("Train Neural Network", type="primary"):
        with st.spinner(f"Training for up to {epochs} epochs..."):
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            
            history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=32, 
                                verbose=0, class_weight=class_weights, callbacks=[early_stopping, reduce_lr])
            
            st.session_state['nn_history_churn'] = history.history
            st.session_state['nn_model_churn'] = model
            st.success(f"Training Complete! Stopped after {len(history.epoch)} epochs.")

    if 'nn_history_churn' in st.session_state:
        history = st.session_state['nn_history_churn']
        model = st.session_state['nn_model_churn']
        
        st.subheader("📈 Training History")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        plt.style.use('dark_background')
        ax1.plot(history['accuracy'], label='Train Accuracy'); ax1.plot(history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy'); ax1.legend()
        ax2.plot(history['loss'], label='Train Loss'); ax2.plot(history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss'); ax2.legend()
        st.pyplot(fig, clear_figure=True)

        st.subheader("🧪 Final Performance on Test Data")
        y_pred_nn = (model.predict(X_test) > 0.5).astype("int32")
        report_nn = classification_report(y_test, y_pred_nn, target_names=y_labels, output_dict=True)
        st.dataframe(pd.DataFrame(report_nn).transpose().style.format("{:.2f}"))

# ---
# LightGBM Page
# ---
def page_lightgbm(X_train, X_test, y_train, y_test, features, y_labels, use_smote):
    st.title("💡 LightGBM for Churn Prediction")
    go_back_button()
    st.markdown(f"Training a LightGBM classifier. This model is using {'SMOTE' if use_smote else '`scale_pos_weight`'} to handle imbalanced data.")

    st.sidebar.header("Hyperparameter Tuning")
    n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 150, 50)
    max_depth = st.sidebar.slider("Max Depth", 2, 20, 7, 1)
    learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2], value=0.1)

    scale_pos_weight = None if use_smote else (y_train.value_counts()[0] / y_train.value_counts()[1])
    
    model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                               random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📊 Model Performance")
    report = classification_report(y_test, y_pred, target_names=y_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Feature Importance")
        importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=True)
        fig = px.bar(importances.tail(15), x='importance', y='feature', orientation='h', title="Top 15 Feature Importances", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🌀 Confusion Matrix")
        with plt.style.context('dark_background'):
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y_labels, yticklabels=y_labels, ax=ax)
            ax.set_title('Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            st.pyplot(fig, clear_figure=True)

# ---
# CatBoost Page
# ---
def page_catboost(X_train, X_test, y_train, y_test, features, y_labels, use_smote):
    st.title("🐈 CatBoost for Churn Prediction")
    go_back_button()
    st.markdown(f"Training a CatBoost classifier. This model is using {'SMOTE' if use_smote else '`scale_pos_weight`'} to handle imbalanced data.")

    st.sidebar.header("Hyperparameter Tuning")
    iterations = st.sidebar.slider("Iterations", 50, 500, 200, 50)
    depth = st.sidebar.slider("Depth", 2, 10, 6, 1)
    learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2], value=0.1)

    scale_pos_weight = None if use_smote else (y_train.value_counts()[0] / y_train.value_counts()[1])
    
    model = cb.CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate,
                                  random_state=42, scale_pos_weight=scale_pos_weight, verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📊 Model Performance")
    report = classification_report(y_test, y_pred, target_names=y_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Feature Importance")
        importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=True)
        fig = px.bar(importances.tail(15), x='importance', y='feature', orientation='h', title="Top 15 Feature Importances", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🌀 Confusion Matrix")
        with plt.style.context('dark_background'):
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=y_labels, yticklabels=y_labels, ax=ax)
            ax.set_title('Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            st.pyplot(fig, clear_figure=True)

# ---
# XGBoost Page
# ---
def page_xgboost(X_train, X_test, y_train, y_test, features, y_labels, use_smote):
    st.title("🚀 XGBoost for Churn Prediction")
    go_back_button()
    st.markdown(f"Training an XGBoost classifier. This model is using {'SMOTE' if use_smote else '`scale_pos_weight`'} to handle imbalanced data.")

    st.sidebar.header("Hyperparameter Tuning")
    n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 150, 50)
    max_depth = st.sidebar.slider("Max Depth", 2, 20, 5, 1)
    learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2], value=0.1)

    scale_pos_weight = None if use_smote else (y_train.value_counts()[0] / y_train.value_counts()[1])
    
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                              random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📊 Model Performance")
    report = classification_report(y_test, y_pred, target_names=y_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Feature Importance")
        importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=True)
        fig = px.bar(importances.tail(15), x='importance', y='feature', orientation='h', title="Top 15 Feature Importances", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🌀 Confusion Matrix")
        with plt.style.context('dark_background'):
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=y_labels, yticklabels=y_labels, ax=ax)
            ax.set_title('Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            st.pyplot(fig, clear_figure=True)

if __name__ == "__main__":
    main()