import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from numpy.random import default_rng
import json
import warnings
warnings.filterwarnings("ignore")

regions = open('regions.json')
region_data = json.load(regions)
final_data = pd.DataFrame(columns=['Country', 'CountryCode', 'LogisticRegression', 'RandomForest', 'KNN',
                                   'DecisionTree', 'LinearSupportVector', 'XGBOOST', 'GaussianNaiveBayes', 'AdaBoost'])

for region in region_data:
    if region["id"] == 'global':
        continue

    model_dict = {}

    print("Running for the country: ", region["value"])
    model_dict['Country'] = region["value"]
    model_dict['CountryCode'] = region["id"]

    dataframe = pd.read_csv("countryData/region_" + region["id"].upper())
    dataframe = dataframe.drop('year', axis=1)
    dataframe = dataframe.dropna()

    # Dropping number of hits to maintain realistic hit-miss ratio
    size_of_hits = sum(dataframe['hit'] == 1)
    size_to_drop = int((size_of_hits * 75)/100)
    arr_indices_top_drop = default_rng().choice(dataframe[dataframe['hit'] == 1].uri, size=size_to_drop, replace=False)
    dataframe = dataframe[~dataframe.uri.isin(arr_indices_top_drop)]

    features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "liveness", "loudness",
                "speechiness", "tempo", "valence"]

    training = dataframe.sample(frac=0.8, random_state=420)
    X_train = training[features]
    y_train = training['hit']
    X_test = dataframe.drop(training.index)[features]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=420)

    # Logistic Regression
    LogisticRegression_Model = LogisticRegression()
    LogisticRegression_Model.fit(X_train, y_train)
    LogisticRegression_Predict = LogisticRegression_Model.predict(X_valid)
    LogisticRegression_Accuracy = accuracy_score(y_valid, LogisticRegression_Predict)
    model_dict['LogisticRegression'] = LogisticRegression_Accuracy

    # Random Forest
    RandomForest_Model = RandomForestClassifier()
    RandomForest_Model.fit(X_train, y_train)
    RandomForest_Predict = RandomForest_Model.predict(X_valid)
    RandomForest_Accuracy = accuracy_score(y_valid, RandomForest_Predict)
    model_dict['RandomForest'] = RandomForest_Accuracy

    # KNN
    KNN_Model = KNeighborsClassifier()
    KNN_Model.fit(X_train, y_train)
    KNN_Predict = KNN_Model.predict(X_valid)
    KNN_Accuracy = accuracy_score(y_valid, KNN_Predict)
    model_dict['KNN'] = KNN_Accuracy

    # Decision Tree
    DecisionTree_Model = DecisionTreeClassifier()
    DecisionTree_Model.fit(X_train, y_train)
    DecisionTree_Predict = DecisionTree_Model.predict(X_valid)
    DecisionTree_Accuracy = accuracy_score(y_valid, DecisionTree_Predict)
    model_dict['DecisionTree'] = DecisionTree_Accuracy

    # Linear Support Vector
    training_LinearSupportVector = training.sample(3000)
    X_train_LinearSupportVector = training_LinearSupportVector[features]
    y_train_LinearSupportVector = training_LinearSupportVector['hit']
    X_test_LinearSupportVector = dataframe.drop(training_LinearSupportVector.index)[features]
    X_train_LinearSupportVector, X_valid_LinearSupportVector, y_train_LinearSupportVector, y_valid_LinearSupportVector \
        = train_test_split(X_train_LinearSupportVector, y_train_LinearSupportVector, test_size=0.2, random_state=420)

    LinearSupportVector_Model = LinearSVC()
    LinearSupportVector_Model.fit(X_train_LinearSupportVector, y_train_LinearSupportVector)
    LinearSupportVector_Predict = LinearSupportVector_Model.predict(X_valid_LinearSupportVector)
    LinearSupportVector_Accuracy = accuracy_score(y_valid_LinearSupportVector, LinearSupportVector_Predict)
    model_dict['LinearSupportVector'] = LinearSupportVector_Accuracy

    # XGBOOST
    XGBoost_Model = XGBClassifier(objective="binary:logistic", n_estimators=10, seed=123)
    XGBoost_Model.fit(X_train, y_train)
    XGBoost_Predict = XGBoost_Model.predict(X_valid)
    XGBoost_Accuracy = accuracy_score(y_valid, XGBoost_Predict)
    model_dict['XGBOOST'] = XGBoost_Accuracy

    # Gaussian Naive Bayes
    GaussianNaiveBayes_Model = GaussianNB()
    GaussianNaiveBayes_Model.fit(X_train, y_train)
    GaussianNaiveBayes_Predict = GaussianNaiveBayes_Model.predict(X_valid)
    GaussianNaiveBayes_Accuracy = accuracy_score(y_valid, GaussianNaiveBayes_Predict)
    model_dict['GaussianNaiveBayes'] = GaussianNaiveBayes_Accuracy

    # AdaBoost
    AdaBoost_Model = AdaBoostClassifier()
    AdaBoost_Model.fit(X_train, y_train)
    AdaBoost_Predict = AdaBoost_Model.predict(X_valid)
    AdaBoost_Accuracy = accuracy_score(y_valid, AdaBoost_Predict)
    model_dict['AdaBoost'] = AdaBoost_Accuracy

    final_data = final_data.append(model_dict, ignore_index=True)

    # Pickle
    pickle.dump(LogisticRegression_Model, open("model/LogisticRegression/" + region["id"]+".pkl", "wb"))
    pickle.dump(RandomForest_Model, open("model/RandomForest/" + region["id"]+".pkl", "wb"))
    pickle.dump(KNN_Model, open("model/KNN/" + region["id"]+".pkl", "wb"))
    pickle.dump(DecisionTree_Model, open("model/DecisionTree/" + region["id"]+".pkl", "wb"))
    pickle.dump(LinearSupportVector_Model, open("model/LinearSupportVector/" + region["id"]+".pkl", "wb"))
    pickle.dump(XGBoost_Model, open("model/XGBoost/" + region["id"]+".pkl", "wb"))
    pickle.dump(GaussianNaiveBayes_Model, open("model/GaussianNaiveBayes/" + region["id"]+".pkl", "wb"))
    pickle.dump(AdaBoost_Model, open("model/AdaBoost/" + region["id"]+".pkl", "wb"))

# Save the data
final_data.to_csv('country_wise_accuracy.csv', index=False)
