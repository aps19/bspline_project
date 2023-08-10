import os
import sys
sys.path.append('/home/abhishek/datascience/end-to-end/BSplineRegression_Model/src/')

from exception import CustomException

from exception import CustomException
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



from components.data_transformation import DataTransformer
from components.data_transformation import DataTransformerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv') # This is the path where the train data will be stored
    test_data_path: str=os.path.join('artifacts', 'test.csv') # Path where the test data will be stored
    raw_data_path: str=os.path.join('artifacts', 'data.csv') # Path where raw data will be stored
    
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        
        try:
            df = pd.read_csv('notebook/B-Spline-Regression-main/HousingData.csv')
            logging.info("Read the dataset as pandas dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved at {}".format(self.ingestion_config.raw_data_path))
            
            logging.info("Splitting the data into train and test")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) 
            logging.info("Train and test data saved at {} and {}".format(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path))
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.initiate_data_ingestion()
    
    data_transformer = DataTransformer()
    data_transformer.initiate_data_transformer(train_data,test_data)