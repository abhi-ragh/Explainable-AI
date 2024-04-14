import tkinter as tk
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('TkAgg')  # Ensures compatibility with Tkinter
import matplotlib.pyplot as plt
import shap
from PIL import Image, ImageTk

# Load the training data
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Select features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Train a random forest model on the full dataset
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(home_data[features], y)

predicted_price_label = None  # Define the global label variable outside the function

def predict_price(user_features):
    """Predicts the house price based on user input.

    Args:
        user_features (dict): A dictionary containing feature names as keys and user-entered values as values.

    Returns:
        float: The predicted house price.
    """

    user_data = pd.DataFrame(user_features, index=[0])

    # Make prediction on user data
    predicted_price = rf_model.predict(user_data)[0]

    global predicted_price_label  # Declare global to modify inside the function
    if predicted_price_label:
        predicted_price_label.config(text=f"Predicted Price: ${predicted_price:.2f}")  # Update the existing label
    else:
        predicted_price_label = tk.Label(root, text=f"Predicted Price: ${predicted_price:.2f}")
        predicted_price_label.pack(pady=10)  # Pack only if the label doesn't exist

    return predicted_price


def explain_prediction(user_data):
    """Explains the predictions using SHAP.

    Args:
        user_data (pd.DataFrame): A DataFrame containing user-entered features.

    Returns:
        None
    """

    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(user_data)

    # Define the SHAP plot function
    def shap_plot():
        plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization
        shap.force_plot(explainer.expected_value, shap_values[0], user_data, matplotlib=True)

    # Call the SHAP plot function within a new thread
    # to prevent blocking the GUI
    import threading
    shap_thread = threading.Thread(target=shap_plot)
    shap_thread.start()

def predict_and_explain():
    user_features = {}
    for feature, entry_field in user_input_fields.items():
        try:
            user_features[feature] = float(entry_field.get())
        except ValueError:
            print(f"Invalid value for {feature}. Please enter a number.")
            return  # Exit the function if there's an invalid input

    # Call the predict_price function to get the predicted price and label
    predicted_price = predict_price(user_features)

    def shap_plot():
        user_data = pd.DataFrame(user_features, index=[0])
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(user_data)

        shap.force_plot(explainer.expected_value, shap_values[0], user_data, matplotlib=True)

        # Clear the Matplotlib figure to avoid memory leaks
        plt.close()

    # Call the shap_plot function within the main thread using `after_idle`
    root.after_idle(shap_plot)



# Create the Tkinter GUI
root = tk.Tk()
root.title("House Price Prediction")

# Heading label
heading_label = tk.Label(root, text="HOUSE PRICE PREDICTION", font=("Arial", 18))
heading_label.pack(pady=10)

# Input fields and labels
user_input_fields = {}
for feature in features:
    feature_label = tk.Label(root, text=feature)
    feature_label.pack()

    entry_field = tk.Entry(root)
    entry_field.pack()

    user_input_fields[feature] = entry_field

# Predict button
predict_button = tk.Button(root, text="Predict Price", command=predict_and_explain)
predict_button.pack(pady=10)


root.mainloop()  # Start the GUI event loop