import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import shap

print()
print('''██   ██  ██████  ██    ██ ███████ ███████     ██████  ██████  ██  ██████ ███████     ██████  ██████  ███████ ██████  ██  ██████ ████████  ██████  ██████  
██   ██ ██    ██ ██    ██ ██      ██          ██   ██ ██   ██ ██ ██      ██          ██   ██ ██   ██ ██      ██   ██ ██ ██         ██    ██    ██ ██   ██ 
███████ ██    ██ ██    ██ ███████ █████       ██████  ██████  ██ ██      █████       ██████  ██████  █████   ██   ██ ██ ██         ██    ██    ██ ██████  
██   ██ ██    ██ ██    ██      ██ ██          ██      ██   ██ ██ ██      ██          ██      ██   ██ ██      ██   ██ ██ ██         ██    ██    ██ ██   ██ 
██   ██  ██████   ██████  ███████ ███████     ██      ██   ██ ██  ██████ ███████     ██      ██   ██ ███████ ██████  ██  ██████    ██     ██████  ██   ██ 
                                                                                                                                                         ''')
print()

# Load the training data
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Select features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Train a random forest model on the full dataset
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(home_data[features], y)

# Get user input for features
user_features = {}
for feature in features:
  value = float(input(f"Enter value for {feature}: "))
  user_features[feature] = value

# Convert user input to a pandas DataFrame
user_data = pd.DataFrame(user_features, index=[0])

# Make prediction on user data
predicted_price = rf_model.predict(user_data)[0]
print(f"Predicted Price: ${predicted_price:.2f}")


# Explain predictions using SHAP
explainer = shap.TreeExplainer(rf_model)
instance_to_explain = user_data

shap_values = explainer.shap_values(instance_to_explain)

# Define the shap_plot function (assuming you want a force plot)
def shap_plot():
    p = shap.force_plot(explainer.expected_value, shap_values[0], user_data, matplotlib=True)
    plt.show()

# Call the shap_plot function
shap_plot()

# Print the predicted price

