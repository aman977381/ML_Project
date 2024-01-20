import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd 



from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import evaluate_models,save_object

import mlflow
from urllib.parse import urlparse

import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)

        return rmse,mae,r2
            
    def initiate_model_trainer(self,train_array,test_array):
        try:
            print("Model training has been initiated \n")
            logging.info('Split Training and Test input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            
            model = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regressor": LinearRegression(),
                "Ridge": Ridge(),
                "XGBoost Regressor": XGBRFRegressor(),
                "KNeighbors Regressor":KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                'Decision Tree':{
                    #'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'max_depth':[2,5,7,10],
                    'min_samples_split':[2,3,4,5],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },
                'Random Forest':{
                    #'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'max_features':['sqrt','log2'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'Gradient Boosting':{
                    'learning_rate':[.1,.01,.05,.08,.085,.09],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
                },
                'Linear Regressor':{},
                'Ridge':{},
                'XGBoost Regressor':{
                    'max_depth':[2,6,8,10],
                    'learning_rate':[.1,.01,.05,.09],
                    'reg_alpha':[1,2,3,5,7,10],
                    'reg_lambda':[1,2,3,5,7,10]
                },
                'AdaBoost Regressor':{
                    'learning_rate':[.1,0.1,0.5,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'KNeighbors Regressor':{
                    'n_neighbors':[4,5,6,8,10],
                    'weights':['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [1,5,10,20,30,40,50]
                }

            }

            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,model,params)

            print("Model training is completed \n")
            ## To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model  = model[best_model_name]

            print(f"This is the best model: {best_model_name}")

            model_name = list(params.keys())
            actual_model = ""

            for model in model_name:
                if best_model_name == model:
                    actual_model = actual_model+model

            best_params = params[actual_model]


            mlflow.set_registry_uri("https://dagshub.com/aman977381/ML_Project.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            ## MLFlow 
            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)

                (rmse,mae,r2) = self.eval_metrics(y_test,predicted_qualities)

                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2",r2)
                mlflow.log_metric("mae",mae)

                if tracking_url_type_store!="file":
                    mlflow.sklearn.log_model(best_model,"model",registered_model_name=actual_model)

                else:
                    mlflow.sklearn.log_model(best_model,"model")


            if best_model_score<0.6:
                raise CustomException(f"No best model found: {best_model_score}")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model 
            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test,predicted)
            return best_model,score

        except Exception as e:
            raise CustomException(e,sys)