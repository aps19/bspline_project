import sys
import pandas as pd
import numpy as np
from exception import CustomException
from logger import logging
from utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:    
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(filepath=model_path)
            preprocessor=load_object(filepath=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,
                crime_rate:float,   
                residential_land_zone:float,
                tract_bounds:float,
                num_of_rooms:float,
                age_of_building:float,
                radial_highways_accessibility:int,
                tax_rate:int,
                pupil_teacher_ratio:float,
                lower_status_population:float):
            
            self.crime_rate=crime_rate
            self.residential_land_zone=residential_land_zone
            self.tract_bounds=tract_bounds
            self.num_of_rooms=num_of_rooms
            self.age_of_building=age_of_building
            self.radial_highways_accessibility=radial_highways_accessibility
            self.tax_rate=tax_rate
            self.pupil_teacher_ratio=pupil_teacher_ratio
            self.lower_status_population=lower_status_population

    def get_data_as_datframe(self):
        try:
            data=pd.DataFrame({
                'CRIM':self.crime_rate,
                'ZN':self.residential_land_zone,
                'INDUS':self.tract_bounds,
                'CHAS':0,
                'NOX':0,
                'RM':self.num_of_rooms,
                'AGE':self.age_of_building,
                'DIS':0,
                'RAD':self.radial_highways_accessibility,
                'TAX':self.tax_rate,
                'PTRATIO':self.pupil_teacher_ratio,
                'B':0,
                'LSTAT':self.lower_status_population
            },index=[0])
            return data
        except Exception as e:
            raise CustomException(e,sys)