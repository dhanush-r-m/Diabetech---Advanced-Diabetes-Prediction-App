import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap
import pickle
import requests
import streamlit_lottie as st_lottie

# Load the dataset
dataset = pd.read_csv("diabetes.csv")
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='median')
dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])
X = dataset.drop(["Outcome"], axis=1)
y = dataset["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# Custom CSS for Night Theme
night_theme = """
    <style>
    body {
        background-color: #0e1117;
        color: #cfd2d6;
    }
    .stApp {
        background-color: #0e1117;
        color: #cfd2d6;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #cfd2d6;
    }
    .stButton button {
        background-color: #1f77b4;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #161b22;
    }
    </style>
"""
st.markdown(night_theme, unsafe_allow_html=True)

# Load Lottie animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_diabetes = load_lottie_url("https://lottie.host/43d3d0fb-9f7e-46dc-9c18-355cfd7836f1/dB6JnNNwE2.json")

st.title("🩺 Diabetes Prediction App")
st.markdown("""
    Welcome to the advanced version of the Diabetes Prediction App! This version allows you to choose from multiple algorithms, 
    tune hyperparameters, and get detailed model performance insights.
""")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("🔧 Model Settings")
    st_lottie.st_lottie(lottie_diabetes, height=200, width=200)

    model_choice = st.selectbox("Select Model", ["K-Nearest Neighbors", "Logistic Regression", "Linear Regression", 
                                                 "Random Forest", "Gradient Boosting", "SVM", "XGBoost"])
    
    if model_choice == "K-Nearest Neighbors":
        n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    elif model_choice == "Logistic Regression":
        c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=c_value, random_state=42)
    
    elif model_choice == "Linear Regression":
        model = LinearRegression()
    
    elif model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth of Trees", 2, 20, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    elif model_choice == "Gradient Boosting":
        learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
        n_estimators = st.slider("Number of Estimators", 50, 200, 100)
        model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
    
    elif model_choice == "SVM":
        c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
        model = SVC(C=c_value, kernel=kernel, probability=True, random_state=42)
    
    elif model_choice == "XGBoost":
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        n_estimators = st.slider("Number of Estimators", 50, 200, 100)
        max_depth = st.slider("Max Depth of Trees", 3, 10, 5)
        model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    show_dataset = st.checkbox("Show Dataset Overview", value=True)
    show_pairplot = st.checkbox("Show Pairplot Visualization", value=False)
    show_performance = st.checkbox("Show Model Performance", value=True)
    show_advanced_metrics = st.checkbox("Show Advanced Metrics (F1-Score, AUC-ROC)", value=True)
    show_shap = st.checkbox("Show SHAP Visualizations", value=False)

# Train the selected model
with st.spinner('Training the model...'):
    model.fit(X_train_poly, y_train)

# Save the model
pickle_out = open("classifier.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

# Prediction form
with st.form("prediction_form"):
    st.subheader("🔍 Enter Patient's Health Metrics")
    Pregnancies = st.number_input("Pregnancies", min_value=0, value=1, step=1)
    Glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0, step=1.0)
    BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0, step=1.0)
    SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0, step=1.0)
    Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, value=80.0, step=1.0)
    BMI = st.number_input("BMI", min_value=0.0, value=30.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
    Age = st.number_input("Age", min_value=1, value=30, step=1)
    submit_button = st.form_submit_button(label='🌟 Predict Diabetes')

if submit_button:
    if Glucose <= 0:
        st.warning("Glucose level cannot be zero or negative.")
    else:
        input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], 
                                  columns=X.columns)
        input_data = poly.transform(scaler.transform(input_data))
        prediction = model.predict(input_data)
        
        if model_choice == "Linear Regression":
            prediction = [1 if p > 0.5 else 0 for p in prediction]  
        
        result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
        st.success(f"🎉 Prediction: **{result}**")

# Show dataset overview and pairplot
if show_dataset:
    st.markdown("### 📊 Dataset Overview & Visualization")
    st.write(dataset.head())

if show_pairplot:
    st.markdown("#### 🌸 Pairplot of Diabetes Features")
    sns.pairplot(dataset, hue="Outcome", palette="coolwarm")
    st.pyplot(plt)

# Model performance evaluation
if show_performance:
    st.markdown("### 📈 Model Performance")

    y_pred = model.predict(X_test_poly)
    if model_choice == "Linear Regression":
        y_pred = [1 if p > 0.5 else 0 for p in y_pred]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write("**Classification Report:**")
    st.json(report)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# Advanced metrics: F1 Score and AUC-ROC
if show_advanced_metrics:
    st.markdown("### 📊 Advanced Metrics")

    f1 = f1_score(y_test, y_pred)
    st.markdown(f"**F1 Score:** {f1:.2f}")

    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test_poly)[:, 1]
    else:
        y_pred_prob = (model.predict(X_test_poly) - model.predict(X_test_poly).min()) / \
                      (model.predict(X_test_poly).max() - model.predict(X_test_poly).min())
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = auc(fpr, tpr)

    st.markdown(f"**AUC-ROC Score:** {auc_score:.2f}")

    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name='AUC-ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='grey')))
    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      showlegend=False)
    st.plotly_chart(fig)

# SHAP Visualizations
if show_shap:
    st.markdown("### 🌟 SHAP Visualizations")
    if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
            shap_values = explainer.shap_values(X_test_poly)
            st.markdown("#### Summary Plot")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test_poly, feature_names=poly.get_feature_names_out(X.columns), show=False)
            st.pyplot(fig)
            st.markdown("#### Feature Importance")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test_poly, feature_names=poly.get_feature_names_out(X.columns), plot_type="bar", show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred with SHAP visualizations: {e}")
    else:
        st.markdown("SHAP explanations are only available for tree-based models.")

# Developer Tag with GitHub icon
st.markdown(
    """
    ---
    <div style='text-align: center;'>
        <h4>👨‍💻 Developed by <strong>Dhanush Moolemane</strong></h4>
        <a href="https://github.com/dhanush-r-m" target="_blank">
            <img src="https://img.icons8.com/ios-glyphs/30/ffffff/github.png"/>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("Thank you for using the **Diabetes Prediction App**! Stay healthy and take care.")
