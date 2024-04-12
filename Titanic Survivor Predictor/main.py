import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import webbrowser
import warnings
warnings.filterwarnings("ignore")

print()
print('''███████ ██    ██ ██████  ██    ██ ██ ██    ██  ██████  ██████      ██████  ██████  ███████ ██████  ██  ██████ ████████  ██████  ██████  
██      ██    ██ ██   ██ ██    ██ ██ ██    ██ ██    ██ ██   ██     ██   ██ ██   ██ ██      ██   ██ ██ ██         ██    ██    ██ ██   ██ 
███████ ██    ██ ██████  ██    ██ ██ ██    ██ ██    ██ ██████      ██████  ██████  █████   ██   ██ ██ ██         ██    ██    ██ ██████  
     ██ ██    ██ ██   ██  ██  ██  ██  ██  ██  ██    ██ ██   ██     ██      ██   ██ ██      ██   ██ ██ ██         ██    ██    ██ ██   ██ 
███████  ██████  ██   ██   ████   ██   ████    ██████  ██   ██     ██      ██   ██ ███████ ██████  ██  ██████    ██     ██████  ██   ██ ''')
print()

import lime
import lime.lime_tabular

data = pd.read_csv("train.csv")
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])

train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)

train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)

train = train.drop(['PassengerId'], axis=1)
test = test.drop(['PassengerId'], axis=1)

# Convert categorical variables into dummy/indicator variables
train_processed = pd.get_dummies(train)
test_processed = pd.get_dummies(test)

# Filling Null Values
train_processed = train_processed.fillna(train_processed.mean())
test_processed = test_processed.fillna(test_processed.mean())

# Create X_train,Y_train,X_test
X_train = train_processed.drop(['Survived'], axis=1)
Y_train = train_processed['Survived']

X_test  = test_processed.drop(['Survived'], axis=1)
Y_test  = test_processed['Survived']

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest_preds = random_forest.predict(X_test)

predict_fn_rf = lambda x: random_forest.predict_proba(x).astype(float)
X = X_train.values
explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = X_train.columns,class_names=['Will Die','Will Survive'],kernel_width=5)


#choosen_instance = X_test.loc[[421]].values[0]
#exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)

#exp.save_to_file('lime_explanation.html')


# Create a function to preprocess user input
def preprocess_user_input(user_input):
    # Convert user input to a DataFrame with the same columns as X_train
    user_df = pd.DataFrame([user_input])
    user_df = pd.get_dummies(user_df)  # Convert categorical variables to dummy variables
    user_df = user_df.reindex(columns=X_train.columns, fill_value=0)  # Reindex to match X_train columns
    return user_df

# Create a function to predict and explain
def predict_and_explain(user_input):
    preprocessed_input = preprocess_user_input(user_input)
    prediction = random_forest.predict(preprocessed_input)
    explanation = explainer.explain_instance(preprocessed_input.values[0], predict_fn_rf, num_features=10)
    return prediction, explanation


# Example usage:
if __name__ == "__main__":
    sex = input("Sex (M/F): ")
    age = int(input("Age: "))
    fare = int(input("Fare: "))
    pclass = int(input("Passenger Class(1,2,3): "))
    sibsp = int(input("No of Siblings/Spouses Boarded: "))
    embarked = input("Embarked From [C for Cherbourg, Q for Queenstown, S for Southampton]: ")
    parch = int(input("No of Parents/Children Boarded: "))

    user_input = {
        "Sex_female": 1 if sex=='F' else 0,
        "Sex_male": 1 if sex=='M' else 0,
        "Age": age,
        "Fare": fare,
        "Pclass": pclass,
        "SibSp": sibsp,
        "Embarked_C": 1 if embarked=='C' else 0,
        "Embarked_Q": 1 if embarked=='Q' else 0,
        "Embarked_S": 1 if embarked=='S' else 0,
        "Parch": parch,
    }

    prediction, explanation = predict_and_explain(user_input)

    print(f"Prediction: {'Will Survive' if prediction[0] == 1 else 'Will Die'}")

    explanation.save_to_file('lime_explanation.html')
    webbrowser.open('lime_explanation.html')