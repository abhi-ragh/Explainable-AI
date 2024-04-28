import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular

def predict_survivor(user_input):
    data = pd.read_csv("Titanic_Survivor_Predictor/train.csv")
    train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['Survived'])

    train = train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    test = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    train_processed = pd.get_dummies(train)
    test_processed = pd.get_dummies(test)

    train_processed = train_processed.fillna(train_processed.mean())
    test_processed = test_processed.fillna(test_processed.mean())

    X_train = train_processed.drop(['Survived'], axis=1)
    Y_train = train_processed['Survived']
    
    X_test = test_processed.drop(['Survived'], axis=1)
    Y_test = test_processed['Survived']

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)

    predict_fn_rf = lambda x: random_forest.predict_proba(x).astype(float)
    X = X_train.values
    explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=X_train.columns,
                                                       class_names=['Will Die', 'Will Survive'], kernel_width=5)

    preprocessed_input = preprocess_user_input(user_input)
    prediction = random_forest.predict(preprocessed_input)
    explanation = explainer.explain_instance(preprocessed_input.values[0], predict_fn_rf, num_features=10)
    lime_html = explanation.as_html()

    return prediction, lime_html

def preprocess_user_input(user_input):
    # Ensure that the 'Sex' key is present in the user input dictionary
    #sex = user_input.get('Sex', 'Unknown')
    # Create a DataFrame from the user input dictionary
    user_df = pd.DataFrame([user_input])
    # Encode 'Sex' feature into separate binary columns
    #user_df['Sex_female'] = 1 if sex == 'Female' else 0
    #user_df['Sex_male'] = 1 if sex == 'Male' else 0
    # Check if 'Sex' column exists before dropping it
    if 'Sex' in user_df.columns:
        user_df.drop('Sex', axis=1, inplace=True)
    # Reorder columns to match the order of features used during training
    user_df = user_df.reindex(columns=[ 'Pclass', 'Age','SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S'], fill_value=0)
    return user_df
