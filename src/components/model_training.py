import os
import sys
import numpy as np
from dataclasses import dataclass


from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    AdaBoostRegressor 
)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from exception import CustomException
from logger import logging
from utils import (
    save_object,
    evaluate_model
)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], 
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighbors": KNeighborsRegressor(),
                "CatBoost": CatBoostRegressor(),
                "XGBoost": XGBRegressor()
            }

            model_report:dict=evaluate_model(models=models,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
            ## To get the best model score from dictionary
            best_model_score = max(model_report.items(), key=lambda x: x[1]['r2'])
            
            # To get the best model name
            best_model_name = best_model_score[0]
            
            best_model = models[best_model_name]

            # If best_model_score is less than 0.6 then raise exception
            if best_model_score[1]['r2'] < 0.6:
                raise CustomException('No model is good enough',sys)
            
            logging.info(f'Best model on the basis of r2 score is {best_model_name}')
            
            save_object(filepath=self.model_trainer_config.trained_model_file_path,
                        obj=best_model
            
            )
            
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)