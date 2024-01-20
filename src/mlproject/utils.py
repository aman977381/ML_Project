import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import numpy 
import pickle
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    logging.info("Reading SQL database started")
    try :
        mydb = pymysql.connect(
            host = host,
            user = user,
            password = password,
            db = db
        )
        logging.info("Connection stablished",mydb)
        df = pd.read_sql_query('Select * from student',mydb)
        print(df.head(5))

        return df

    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        for i in range (len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            print(f"{model} model training has begin \n")
            
            gs = GridSearchCV(model,param_grid=param,cv=5)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            print(f"{model} model training is stoped :")
            print(f"train score is :{train_model_score}, and test score is:{test_model_score}\n")

            report[list(models.keys())[i]] = test_model_score

        return report


    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)