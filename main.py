import streamlit as st
import streamlit.components.v1 as components
from Housing_Price_Prediction.house_price_prediction import predict_house_price
from Titanic_Survivor_Predictor.titanic_survivor_prediction import predict_survivor
import shap
import xgboost

def main():
    st.title("Explainable AI")

    st.write("Choose a prediction task:")
    choice = st.radio("", ("House Price Prediction", "Titanic Survivor Prediction"))

    if choice == "House Price Prediction":
        house_price_page()
    elif choice == "Titanic Survivor Prediction":
        titanic_survivor_page()

def house_price_page():
    st.subheader("House Price Prediction")
    
    st.write("Enter the following details:")
    lot_area = st.number_input("Lot Area")
    year_built = st.number_input("Year Built")
    first_floor_sf = st.number_input("1st Floor Sq Ft")
    second_floor_sf = st.number_input("2nd Floor Sq Ft")
    full_bath = st.number_input("Number of Full Bathrooms")
    bedrooms = st.number_input("Number of Bedrooms")
    total_rooms = st.number_input("Total Rooms Above Ground")

    user_input = {
        "LotArea": lot_area,
        "YearBuilt": year_built,
        "1stFlrSF": first_floor_sf,
        "2ndFlrSF": second_floor_sf,
        "FullBath": full_bath,
        "BedroomAbvGr": bedrooms,
        "TotRmsAbvGrd": total_rooms
    }

    prediction, shap_html = predict_house_price(user_input)

    st.write(f"Predicted Price: ${prediction:.2f}")

    if st.button("Show SHAP Explanation"):
        st.subheader("SHAP Explanation")
        st_shap(shap_html)

def titanic_survivor_page():
    st.subheader("Titanic Survivor Prediction")
    
    st.write("Enter passenger details:")
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=150, step=1)
    fare = st.number_input("Fare", min_value=0.0, step=0.01)
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sibsp = st.number_input("Number of Siblings/Spouses Boarded", min_value=0, step=1)
    embarked = st.selectbox("Embarked From", ["C", "Q", "S"])
    parch = st.number_input("Number of Parents/Children Boarded", min_value=0, step=1)

    user_input = {
        "Sex_female": 1 if sex == "Female" else 0,
        "Sex_male": 1 if sex == "Male" else 0,
        "Age": age,
        "Fare": fare,
        "Pclass": pclass,
        "SibSp": sibsp,
        "Embarked_C": 1 if embarked == "C" else 0,
        "Embarked_Q": 1 if embarked == "Q" else 0,
        "Embarked_S": 1 if embarked == "S" else 0,
        "Parch": parch,
    }

    prediction, lime_html = predict_survivor(user_input)

    st.write(f"Prediction: {'Will Survive' if prediction[0] == 1 else 'Will Die'}")

    if st.button("Show LIME Explanation"):
        st.subheader("LIME Explanation")
        st.components.v1.html(lime_html, height=800)

def st_shap(shap_html):
    shap_html_with_js = f"<head>{shap.getjs()}</head><body><div style='overflow: hidden;margin-top: 100px;padding: 20px;'>{shap_html}</div></body>"
    components.html(shap_html_with_js, height=700, width=1500)



if __name__ == "__main__":
    main()
