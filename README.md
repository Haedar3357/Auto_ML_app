
---

```markdown
# AutoML Streamlit App with AI Assistant

This project is an interactive web application built with **Streamlit** that enables users to upload and analyze datasets, build machine learning models using **PyCaret**, and evaluate model performance—all through a clean visual interface. It also includes an integrated **AI assistant** powered by large language models (LLMs) to help guide users during the process.

## Features

- Built-in AI assistant that answers user questions and provides guidance.
- Comprehensive data profiling using `ydata-profiling`.
- Supports both classification and regression tasks.
- AutoML functionality to compare and select the best model.
- Detailed performance metrics for both training and testing data.
- Download the final trained model in `.pkl` format.

## Application Workflow

### 1. Upload Dataset

Upload a CSV file through the interface. The data is stored locally and used in subsequent steps.

### 2. Data Profiling

Generates an interactive exploratory data analysis (EDA) report to understand distributions, missing values, correlations, and more.

### 3. Modeling

- Select the target column (label).
- Choose task type: **Classification** or **Regression**.
- Option to run all models and compare automatically, or manually select a specific model.
- The trained model is saved locally.

### 4. Model Evaluation

Performance metrics are displayed for both train and test sets:

- **Classification:** Accuracy, F1 Score, Confusion Matrix.
- **Regression:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score.

### 5. Download Model

After training, the best model can be downloaded as a `.pkl` file for reuse in other applications.

### 6. AI Assistant

An interactive assistant powered by an LLM is integrated into the app. Users can ask questions related to the tool, model selection, or data issues.

## Requirements

- Python >= 3.8
- Required Python packages:

  - streamlit  
  - pandas  
  - pycaret  
  - ydata-profiling  
  - streamlit-pandas-profiling  
  - seaborn  
  - matplotlib  
  - requests  
  - streamlit_chat  

Install all dependencies using:

```bash
pip install -r requirements.txt
How to Run the App
Run the application using the following command:

bash
نسخ
تحرير
streamlit run app.py
Notes
To use the AI assistant feature, replace the placeholder API_KEY with your own key from a provider such as OpenRouter.

The best model will be saved locally as best_model.pkl.

License
This project is open-source and available for educational and research purposes.
