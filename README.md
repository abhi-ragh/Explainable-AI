# Explainable-AI

This project empowers humans to understand the "why" behind AI decisions by exploring Explainable AI (XAI) techniques. We demonstrate XAI with two practical models: house price prediction and Titanic survivor prediction.

[Click Here to Visit](https://n0nsense-404-explainable-ai-main-dgjo5b.streamlit.app/)

Project Structure:

    House Price Prediction: Contains datasets used for training and testing the House Price Model
    Titanic Survivor Predicion: Contains datasets used for training and testing the Titanic survival prediction model.
    main: Contains the Streamlit code for building the interactive website.

Getting Started:

    Clone the Repository: Use git clone https://github.com/n0nsense-404/Explainable-AI/ to clone this repository.
    Set Up Environment: Install required dependencies using a virtual environment manager like venv or conda. Refer to the requirements.txt file for specific dependencies.
    Run the Streamlit App: Run streamlit run main.py to launch the website.

The Streamlit website allows users to:

    Input data for house price prediction and receive explanations using SHAP.
    Input features for Titanic survival prediction and get explanations using LIME.
    Visualize the explanations to understand factors influencing the model's predictions.
