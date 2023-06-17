import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
regions = open('regions.json')
region_data = json.load(regions)


@app.route('/predict', methods=['POST'])
def predict():
    input_json = request.get_json(force=True)
    features = [input_json['acousticness'], input_json['danceability'], input_json['duration_ms'], input_json['energy'],
                input_json['instrumentalness'], input_json['liveness'], input_json["loudness"],
                input_json['speechiness'], input_json['tempo'], input_json['valence']]
    final_features = [np.array(features)]

    accuracy = pd.read_csv('country_wise_accuracy.csv')
    performance = pd.DataFrame({'Model': ['LogisticRegression',
                                          'RandomForest',
                                          'KNN',
                                          'DecisionTree',
                                          'LinearSupportVector',
                                          'XGBOOST',
                                          'GaussianNaiveBayes',
                                          'AdaBoost'],
                                'Accuracy': [accuracy["LogisticRegression"].mean(),
                                             accuracy["RandomForest"].mean(),
                                             accuracy["KNN"].mean(),
                                             accuracy["DecisionTree"].mean(),
                                             accuracy["LinearSupportVector"].mean(),
                                             accuracy["XGBOOST"].mean(),
                                             accuracy["GaussianNaiveBayes"].mean(),
                                             accuracy["AdaBoost"].mean()]})

    performance.sort_values(by="Accuracy", ascending=False)
    model1 = performance.iloc[0, 0]
    model2 = performance.iloc[1, 0]
    model3 = performance.iloc[2, 0]

    model1_weight = performance.iloc[0, 1]
    model2_weight = performance.iloc[1, 1]
    model3_weight = performance.iloc[2, 1]

    final_prediction = [["CountryCode", "Country", "HitPrediction"]]
    for region in region_data:
        if region["id"] == 'global':
            continue

        model1_model_weight = pickle.load(open("model/" + model1 + "/" + region["id"] + '.pkl', 'rb'))
        model2_model_weight = pickle.load(open("model/" + model2 + "/" + region["id"] + '.pkl', 'rb'))
        model3_model_weight = pickle.load(open("model/" + model3 + "/" + region["id"] + '.pkl', 'rb'))

        prediction1 = model1_model_weight.predict_proba(final_features)
        output1 = prediction1[0][1]
        prediction2 = model2_model_weight.predict_proba(final_features)
        output2 = prediction2[0][1]
        prediction3 = model3_model_weight.predict_proba(final_features)
        output3 = prediction3[0][1]

        weighted_prediction = (output1 * model1_weight + output2 * model2_weight + output3 * model3_weight) / \
                              (model1_weight + model2_weight + model3_weight)
        weighted_prediction = np.round(weighted_prediction, 2)

        final_prediction.append([region["id"], region["value"], weighted_prediction])

    return jsonify(final_prediction)


if __name__ == "__main__":
    app.run(debug=True)
