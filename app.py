from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/abhishek/datascience/end-to-end/BSplineRegression_Model/src/')

from sklearn.preprocessing import StandardScaler

from pipeline.predictions import PredictPipeline,CustomData

application = Flask(__name__) #Initialize the flask App
app=application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            crime_rate=float(request.form['crime_rate']),
            residential_land_zone=float(request.form['residential_land_zone']),
            tract_bounds=float(request.form['tract_bounds']),
            num_of_rooms=float(request.form['num_of_rooms']),
            age_of_building=float(request.form['age_of_building']),
            radial_highways_accessibility=int(request.form['radial_highways_accessibility']),
            tax_rate=int(request.form['tax_rate']),
            pupil_teacher_ratio=float(request.form['pupil_teacher_ratio']),
            lower_status_population=float(request.form['lower_status_population'])

        )
        
        preds_df = data.get_data_as_datframe()
        print(preds_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(preds_df)
        print(results)
        
        return render_template('home.html',results=results[0])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True) #Run the app in debug mode