from flask import Flask, render_template, request 
import numpy as np 

import pickle 
import joblib 

#initiate the app
app=Flask(__name__)

@app.route('/')
def form():
    return render_template("diabetics.html")

@app.route("/predict", methods=["post"])
def predict():
    v1 = int(request.form.get('Pregnancies'))
    v2 = int(request.form.get('Glucose'))
    v3 = int(request.form.get('BloodPressure'))
    v4 = int(request.form.get('SkinThickness'))
    v5 = int(request.form.get('Insulin'))
    v6 = int(request.form.get('BMI'))
    v7 = int(request.form.get('DiabetesPedigreeFunction'))
    v8 = int(request.form.get('Age'))

    #print("value1", v1)
    # Convert input into array
    input_data = np.array([[v1, v2, v3, v4, v5, v6, v7, v8]])
    
    #loading model and prediction
    model = joblib.load('model_job.pkl')

    prediction= model.predict(input_data)
    prediction1 = int(prediction)
    print(prediction1)

    if prediction1 == 1:
        return "Diabetic"
    else:
        return "Not Diabetics"


    
    return("Form Submitted")
    
#run the code 
app.run(debug=True)
