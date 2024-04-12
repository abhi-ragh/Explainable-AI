import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap

# Load the training data
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Select features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Train a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

# Train a model on the full dataset
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X, y)

# Load test data
test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]

# Make predictions on test data
test_preds = rf_model_on_full_data.predict(test_X)

# Create submission file
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)

# Explain predictions using SHAP
explainer = shap.TreeExplainer(rf_model)
instance_to_explain = test_X.iloc[1]  # Choose a data point from the test set
shap_values = explainer.shap_values(instance_to_explain.values.reshape(1, -1))

# Define the shap_plot function
def shap_plot(j):
    explainerModel = shap.TreeExplainer(rf_model)
    shap_values_Model = explainerModel.shap_values(test_X)
    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], test_X.iloc[[j]], matplotlib=True, show=False)
    plt.savefig('tmp.svg')
    plt.close()
    return p

# Call the shap_plot function
shap_plot(0)  # Replace 1 with the desired index
