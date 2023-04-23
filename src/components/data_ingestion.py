import os
import sys
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

## Intitialize the Data Ingetion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create class for Data Ingetion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingetion method starts")
        try:
            df=pd.read_csv(os.path.join('notebook/data','adult.csv'))
            logging.info('Dataset read as pandas Dataframe')
            df['income'] = df['income'].replace(['<=50K.'], '<=50K')
            df['income'] = df['income'].replace(['>50K.'], '>50K')
            le = LabelEncoder() # label encoder 
            df['income']=le.fit_transform(df['income']) 


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            

            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of Data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingetion stage')
            raise CustomException(e,sys)
        
