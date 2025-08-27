AI Customer Churn Prediction
Overview
This project demonstrates an end-to-end machine learning solution for a business problem. The goal is to build an AI model that predicts customer churn for a telecommunications company. The project includes a complete data science pipeline, from data analysis and model training to an interactive front-end.

The final output is an interactive dashboard built directly in a Jupyter Notebook using the ipywidgets library. This allows users to input customer data via a user-friendly form and get a real-time prediction on the likelihood of that customer churning.

Features
Data Analysis: Exploratory Data Analysis (EDA) to understand key churn drivers.

Machine Learning Model: A robust Random Forest Classifier is used to predict churn.

Interactive Frontend: A user-friendly dashboard with sliders and dropdown menus for real-time predictions.

Complete Pipeline: The project demonstrates a full workflow from raw data to a practical, interactive application.

Technologies Used
Programming Language: Python

Data Manipulation: pandas, numpy

Machine Learning: scikit-learn for preprocessing, modeling, and evaluation.

Interactive UI: ipywidgets for creating the dashboard.

Visualization: matplotlib, seaborn

Environment: Jupyter Notebook or Google Colab

Project Structure
The project code is organized into two main cells within a single Jupyter Notebook file:

Backend (Cell 1): This cell handles all the data science logic. It loads the dataset, cleans it, and trains the machine learning model. This part of the code needs to be run first.

Frontend (Cell 2): This cell creates the interactive user interface using ipywidgets. It defines the form, the button, and the function that takes user input to generate a prediction from the model trained in Cell 1.

How to Run the Project
Download the Dataset: Get the WA_Fn-UseC_-Telco-Customer-Churn.csv file. You can find this on Kaggle.

Open in Jupyter: Open the .ipynb file in a Jupyter Notebook or Google Colab.

Run All Cells: Navigate to the Cell menu and select Run All. This will execute the code sequentially, first training the model and then displaying the interactive dashboard.

Use the Dashboard: After all cells have run, the interactive form will appear. You can adjust the sliders and dropdown menus and click the "Predict Churn" button to see the prediction.

Example Output
The dashboard provides a clear output, displaying the prediction (High or Low Churn Risk) and the exact probability, with a color-coded border for easy interpretation.

Prediction: High Churn Risk

Probability of Churn: 0.78
