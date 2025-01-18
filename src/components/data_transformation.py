import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for Data Transformation
        '''

        try:
            numerical_cols = [
                'Hours Studied',
                'Previous Scores',
                'Sleep Hours',
                'Sample Question Papers Practiced'
            ]

            categorical_cols = ['Extracurricular Activities']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Numerical columns: {numerical_cols}')
            logging.info(f'Categorical columns: {categorical_cols}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info('Reading of train and test data completed')

            logging.info('Obtaining pre-processing object')

            preprocessing_obj = self.get_data_transformer_obj()

            target_feature_name = 'Performance Index'

            input_feature_train_data = train_data.drop(columns=[target_feature_name])
            target_feature_train_data = train_data[target_feature_name]

            input_feature_test_data = test_data.drop(columns=[target_feature_name])
            target_feature_test_data = test_data[target_feature_name]

            logging.info('Applying pre-processing object on training DataFrame and testing DataFrame')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_data)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_data)]

            logging.info('Saved pre-processing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)