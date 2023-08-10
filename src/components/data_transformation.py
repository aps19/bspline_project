import sys
from dataclasses import dataclass

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # for handling missing values
from sklearn.pipeline import Pipeline  # for assembling preprocessing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  

from exception import CustomException
from logger import logging
from utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()
        
    def get_data_transformer(self):
        """
        This method is responsible for returning data transformation.
        
        """
        
        try:
            numerical_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']
            categorical_columns = []
            
            num_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='median')), 
                       ('scaler', StandardScaler())]
            )
                
            cat_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                       ('onehot', OneHotEncoder(handle_unknown='ignore')),
                       ('scaler', StandardScaler())]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
            # logging.info(f"Categorical columns: {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                # ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformer(self, train_path, test_path):
        """
        This method is responsible for initiating data transformation.
        
        """
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading train and test data completed.")
            logging.info("Obtaining data transformer.")
            
            preprocessor = self.get_data_transformer()
            
            target_column = 'MEDV'
            numerical_columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
            
            input_feature_train_df = train_df.drop(columns = [target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns = [target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info(f"Applying data transformation on train and test dataframes.")
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saving preprocessor object.")
            
            save_object(
                        filepath = self.data_transformer_config.preprocessor_file_path,
                        obj = preprocessor
                        )
            
            return (train_arr,
                    test_arr,
                    self.data_transformer_config.preprocessor_file_path)
            
        except Exception as e:
            raise CustomException(e,sys)