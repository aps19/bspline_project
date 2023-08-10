import os
import sys
import numpy as np
import pandas as pd
import dill

from exception import CustomException
from logger import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def save_object(filepath,obj):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(models:dict,X_train:np.ndarray,y_train:np.ndarray,X_test:np.ndarray,y_test:np.ndarray):
    try:
        model_report:dict={}
        for model_name,model in models.items():
            logging.info(f'Training {model_name} model')
            model.fit(X_train,y_train)
            logging.info(f'Predicting {model_name} model')
            y_pred = model.predict(X_test)
            logging.info(f'Calculating metrics for {model_name} model')
            model_report[model_name]={
                'mse':mean_squared_error(y_test,y_pred),
                'mae':mean_absolute_error(y_test,y_pred),
                'r2':r2_score(y_test,y_pred)
            }
        
        return model_report
    
    except Exception as e:
        raise CustomException(e,sys)