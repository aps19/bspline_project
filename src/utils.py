import os
import sys
import numpy as np
import pandas as pd
import dill

from exception import CustomException
from logger import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

def save_object(filepath,obj):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise CustomException(e,sys)

def load_object(filepath):
    try:
        with open(filepath, 'rb') as file:
            obj = dill.load(file)
        return obj
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(models: dict, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, params: dict):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
                        
            logging.info(f'Predicting {list(models.keys())[i]}')

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
