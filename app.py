import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import os
from pycaret.classification import setup as clf_setup, compare_models as clf_compare_models, pull as clf_pull, save_model as clf_save_model, predict_model as clf_predict_model, get_config as clf_get_config, create_model as clf_create_model, finalize_model as clf_finalize_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model, predict_model as reg_predict_model, get_config as reg_get_config, create_model as reg_create_model, finalize_model as reg_finalize_model
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from streamlit_chat import message

# Load dataset if exists
if os.path.exists("dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)

# Sidebar options
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png")
    st.title("AutoML")
    choice = st.radio("Select the task", ["Upload", "Profiling", "Modeling", "Download", "AI Assistant"])
    st.info("This project application helps you build and explore your data")

if choice == "AI Assistant":
    st.title("AI Assistant")

   # Initialize API key variable; should be securely set with a valid key for the AI service
    API_KEY = ""  # Replace with your actual API key or use environment variables for security

    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent assistant inside an AutoML Streamlit app. "
                    "Your role is to help users understand different sections of the app, "
                    "guide them in selecting the right models, explain required data formats, "
                    "and offer tips on model performance and evaluation."
                )
            }
        ]


    for msg in st.session_state["messages"]:
        if msg["role"] != "system":
            message(msg["content"], is_user=(msg["role"] == "user"))

 
    user_input = st.text_input("Ask me anything:")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
    "model": "meta-llama/llama-4-maverick:free",  # نموذج مجاني متاح
    "messages": st.session_state["messages"]
       }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
        else:
            reply = f"Error: {response.text}"

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        message(reply)



# Upload dataset
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df.head(10))

# Data profiling
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile = ydata_profiling.ProfileReport(df, explorative=True)
    st_profile_report(profile)

# Modeling
if choice == "Modeling":
    st.title("Modeling")
    chosen_target = st.selectbox("Select the target column", df.columns)
    st.subheader("Choose the algorithm type")
    algorithm_type = st.radio("Select the algorithm type", ["Classification", "Regression"])
    
    st.subheader("Choose to run a specific model or all models")
    run_all_models = st.radio("Run Model(s)", ["All Models", "Specific Model"])
    
    if algorithm_type == "Classification":
        st.subheader("Classification Algorithms")
        model_options = ['Random Forest', 'K-Nearest Neighbors', 'Naive Bayes', 'SVM', 'Extreme Gradient Boosting', 'Decision Tree Classifier']
        setup = clf_setup
        compare_models = clf_compare_models
        create_model = clf_create_model
        finalize_model = clf_finalize_model
        save_model = clf_save_model
        predict_model = clf_predict_model
        get_config = clf_get_config
        pull = clf_pull
    elif algorithm_type == "Regression":
        st.subheader("Regression Algorithms")
        model_options = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest Regressor', 'Gradient Boosting Regressor', 'Elastic Net']
        setup = reg_setup
        compare_models = reg_compare_models
        create_model = reg_create_model
        finalize_model = reg_finalize_model
        save_model = reg_save_model
        predict_model = reg_predict_model
        get_config = reg_get_config
        pull = reg_pull
    
    if run_all_models == "Specific Model":
        chosen_model = st.selectbox("Select the model to run", model_options)

    if st.button("Run Model(s)"):
        s = setup(data=df, target=chosen_target, normalize=True, verbose=False, html=False, session_id=123)
        
        if run_all_models == "All Models":
            best_model = compare_models()
            save_model(best_model, "best_model")
            model_to_use = best_model
            
            # Display the best model
            st.write(f"Best Model: {model_to_use}")

            # Display the performance metrics
            st.write("Model Performance (Training Data):")
            metrics_df = pull()
            st.dataframe(metrics_df)
        else:
            if chosen_model == 'Random Forest':
                model_to_use = create_model('rf')
            elif chosen_model == 'K-Nearest Neighbors':
                model_to_use = create_model('knn')
            elif chosen_model == 'Naive Bayes':
                model_to_use = create_model('nb')
            elif chosen_model == 'SVM':
                model_to_use = create_model('svm')
            elif chosen_model == 'Extreme Gradient Boosting':
                model_to_use = create_model('xgboost')
            elif chosen_model == 'Decision Tree Classifier':
                model_to_use = create_model('dt')
            elif chosen_model == 'Linear Regression':
                model_to_use = create_model('lr')
            elif chosen_model == 'Ridge Regression':
                model_to_use = create_model('ridge')
            elif chosen_model == 'Lasso Regression':
                model_to_use = create_model('lasso')
            elif chosen_model == 'Random Forest Regressor':
                model_to_use = create_model('rf')
            elif chosen_model == 'Gradient Boosting Regressor':
                model_to_use = create_model('gbr')
            elif chosen_model == 'Elastic Net':
                model_to_use = create_model('en')
                
            model_to_use = finalize_model(model_to_use)
            save_model(model_to_use, f"{chosen_model.replace(' ', '_')}_model")

        # Get train and test data
        X_train = get_config('X_train')
        y_train = get_config('y_train')
        X_test = get_config('X_test')
        y_test = get_config('y_test')
        
        # Get predictions on train and test data
        train_predictions = predict_model(model_to_use, data=X_train)
        test_predictions = predict_model(model_to_use, data=X_test)
        
        # Ensure the column 'Label' exists in predictions
        label_col = 'Label' if 'Label' in test_predictions.columns else 'prediction_label'
        
        if label_col not in test_predictions.columns:
            st.error("Error: Predicted label column not found in predictions.")
            st.stop()

        # Calculate and display metrics for training data
        if algorithm_type == "Classification":
            train_metrics = pd.DataFrame({
                'Accuracy': [accuracy_score(y_train, train_predictions[label_col])],
                'F1 Score': [f1_score(y_train, train_predictions[label_col], average='weighted')],
                # Add other metrics as needed
            })
            st.write("Train Data Performance Metrics:")
            st.dataframe(train_metrics)
            
            # Calculate and display metrics for test data
            test_metrics = pd.DataFrame({
                'Accuracy': [accuracy_score(y_test, test_predictions[label_col])],
                'F1 Score': [f1_score(y_test, test_predictions[label_col], average='weighted')],
                # Add other metrics as needed
            })
            st.write("Test Data Performance Metrics:")
            st.dataframe(test_metrics)
            
            # Calculate confusion matrix for test data if not running all models and it's a classification task
            if run_all_models != "All Models":
                cm_test = confusion_matrix(y_test, test_predictions[label_col])
                
                # Display confusion matrix for test data
                st.write("Confusion Matrix (Test Data):")
                fig, ax = plt.subplots()
                sns.heatmap(cm_test, annot=True, cmap="Blues", fmt="d", ax=ax)
                ax.set_title("Confusion Matrix (Test Data)")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

        elif algorithm_type == "Regression":
            train_metrics = pd.DataFrame({
                'MSE': [mean_squared_error(y_train, train_predictions[label_col])],
                'MAE': [mean_absolute_error(y_train, train_predictions[label_col])],
                'R2 Score': [r2_score(y_train, train_predictions[label_col])],
                # Add other metrics as needed
            })
            st.write("Train Data Performance Metrics:")
            st.dataframe(train_metrics)
            
            # Calculate and display metrics for test data
            test_metrics = pd.DataFrame({
                'MSE': [mean_squared_error(y_test, test_predictions[label_col])],
                'MAE': [mean_absolute_error(y_test, test_predictions[label_col])],
                'R2 Score': [r2_score(y_test, test_predictions[label_col])],
                # Add other metrics as needed
            })
            st.write("Test Data Performance Metrics:")
            st.dataframe(test_metrics)
st.markdown("""
    <style>
    .custom-container {
        background-color: orange;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)
# Download model
if choice == "Download":
    if os.path.exists("best_model.pkl"):
        with open('best_model.pkl', 'rb') as f:
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.download_button('⬇️ Download Model', f, file_name='best_model.pkl')
            st.markdown('</div>', unsafe_allow_html=True)
