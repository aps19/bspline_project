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
            
            params = {
                "RandomForest": {
                    'n_estimators': [8, 16, 32, 64, 100, 200],
                    'max_depth': [2, 4, 6, 8, 10],
                    'max_features': ['sqrt', 'log2', None],
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [2, 4, 6, 8, 10],
                    'max_features': ['sqrt', 'log2', None]
                },
                "GradientBoosting": {
                    'n_estimators': [8, 16, 32, 64, 100, 200],
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8],
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                },
                "AdaBoost": {
                    'n_estimators': [8, 16, 32, 64, 100, 200],
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'loss': ['linear', 'square', 'exponential']
                },
                "LinearRegression": {},
                "KNeighbors": {
                    'n_neighbors': [2, 3, 4, 5, 6, 7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "XGBoost": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [8, 16, 32, 64, 100, 200],
                    'max_depth': [2, 4, 6, 8, 10]
                }
                
                
            }

            model_report:dict=evaluate_model(models=models,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,params=params)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                filepath=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)