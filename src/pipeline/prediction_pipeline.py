import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                age:int,
                workclass:str,
                education_num:int,
                capital_gain:int,
                capital_loss:int,
                hours_per_week:int,
                marital_status:str,
                occupation:str,
                relationship:str, 
                race:str,
                native_country:str,
                sex:str
                ):
        self.age=age
        self.workclass=workclass
        self.education_num=education_num
        self.capital_gain=capital_gain
        self.capital_loss=capital_loss
        self.hours_per_week=hours_per_week
        self.marital_status=marital_status
        self.occupation=occupation
        self.relationship= relationship
        self.race= race
        self.native_country= native_country
        self.sex= sex
        
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':self.age,
                'workClass':self.workclass,
                'education-num':self.education_num,
                'capital-gain':self.capital_gain,
                'capital-loss':self.capital_loss,
                'hours-per-week':self.hours_per_week,
                'marital-status':self.marital_status,
                'occupation':self.occupation,
                'relationship':self.relationship,
                'race':self.race,
                'native-country':self.native_country,
                'sex':self.sex,
                
            }
            df = pd.DataFrame([custom_data_input_dict])
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)