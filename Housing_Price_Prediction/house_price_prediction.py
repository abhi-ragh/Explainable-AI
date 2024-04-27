import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
shap.initjs()

def predict_house_price(user_input):
    home_data = pd.read_csv('Housing_Price_Prediction/train.csv')
    y = home_data['SalePrice']

    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(home_data[features], y)

    user_data = pd.DataFrame(user_input, index=[0])
    predicted_price = rf_model.predict(user_data)[0]

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(user_data)
    shap_html = shap.force_plot(explainer.expected_value, shap_values[0], user_data, matplotlib=False)

    return predicted_price, shap_html._repr_html_()
