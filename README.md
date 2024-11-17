# Heart Disease Prediction

This project uses a Logistic Regression model to predict whether a person has heart disease based on their medical attributes. The model is trained on a dataset and deployed with a simple Streamlit user interface for real-time predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Explanation](#model-explanation)
- [Contributing](#contributing)
- [License](#license)
- [Run](#run)
- [Output](#output)

## Project Overview

The project consists of:
- **Heart Disease Prediction Model**: A Logistic Regression model trained on a dataset with various medical attributes.
- **Streamlit Web Interface**: A user interface that allows users to input their medical data and receive a prediction on whether they are likely to have heart disease or not.

The model predicts the likelihood of heart disease based on the following features:
- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Serum cholesterol (chol)
- Fasting blood sugar (fbs)
- Resting electrocardiographic results (restecg)
- Maximum heart rate achieved (thalach)
- Exercise induced angina (exang)
- ST depression induced by exercise (oldpeak)
- Slope of the peak exercise ST segment (slope)
- Number of major vessels (ca)
- Thalassemia (thal)

## Requirements

Make sure you have the following libraries installed:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit

You can install these requirements using the following command:

```bash
pip install -r requirements.txt
