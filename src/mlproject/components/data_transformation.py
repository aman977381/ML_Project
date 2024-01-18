import sys
import os 
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

from src.mlproject.utils import save_object


@dataclass

class DataTransformationConfig:
    preprocess_pickel_path = os.path.join('artifact','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function  is reponsible for data transformation
        """
        try:
            num_features = ['writing_score','reading_score']
            cat_features = [
                'gender','race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                            ]
            
            num_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('One_Hot_Encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f'categorical columns:{list(cat_features)}')
            logging.info(f'numerical features: {list(num_features)}')

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,num_features),
                ("cat_pipeline",cat_pipeline,cat_features)
            ],remainder='passthrough')

            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test files")

            preprocess_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            num_features = ['writing_score','reading_score']
            cat_features = [
                'gender','race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                            ]
            
            ## divide the train dataset to independent and dependent features
            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_features_train_df = train_df[target_column_name]

            ## divide the test dataset to independent and dependent features
            input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_features_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessing on Train and Test dataframme")

            input_features_train_arr = preprocess_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocess_obj.transform(input_features_test_df)

            ## combined train and test array with target value
            train_arr = np.c_[input_features_train_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_features_test_df)]

            logging.info(f'saved preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocess_pickel_path,
                obj = preprocess_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_pickel_path
            )

        except Exception as e:
            raise CustomException(e,sys)