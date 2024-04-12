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