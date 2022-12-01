from flask import Flask, request, jsonify, Response, render_template
import json
import pickle
import math

app = Flask(__name__)

model = pickle.load(open('src/diabetes_lgbm_model.pkl', 'rb'))
stdScale = pickle.load(open('src/diabetes_stdScale_model.pkl', 'rb'))
list_params = ['Age', 'gender', 'cholestrol', 'check', 'BMI', 'smoker', 'heart', 'activity', 'fruit', 
                'veggies', 'alcohol', 'genhealth', 'mental', 'physical', 'walk', 'hypertension', 'stroke']

@app.route("/", methods=['GET'])
def index():
    return render_template('diabetes_form.html')

@app.route("/results", methods=['POST'])
def output_results(): 
    result = request.form

    X = []
    # build out X to pass into model
    for index in list_params:
        if index == 'Age':
            if float(result[index]) < 18:
                X.append(0.0)
            elif float(result[index]) < 25:
                X.append(1.0)
            else:
                age_5yr = math.ceil((float(result[index]) - 24) / 5)
                X.append(age_5yr)
                
        elif index == 'BMI' or index == 'mental' or index == 'physical' or index == 'genhealth':
            X.append(float(result[index]))
        else:
            if index in result:
                X.append(1.0)
            else:
                X.append(0.0)
    stdScale.transform([X])
    probs = model.predict_proba([X])
    print(probs)
    if probs[0][1] < 0.3:
        output = 'You probably do not have Diabetes'
    elif probs[0][1] < 0.7:
        output = "You might have Diabetes. Please go see a doctor"
    else:
        output = "There is a high likelyhood of you having Diabetes. Please go see a doctor"

    return render_template("results.html", output = output)

if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()