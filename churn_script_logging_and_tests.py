# file level doc string
'''
Author: Matthew Yue
Purpose: This is the logging and testing file for ./churn_library.py
Date: June 23, 2022
'''
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data passes")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
        path = './images/eda'
        destination_dir = os.listdir(path)
        assert len(destination_dir)
        logging.info("SUCCESS: Perform EDA succeeded")
    except AssertionError as err:
        logging.error("ERROR: EDA not successful. Destination folder is empty!")
        raise err

def test_encoder_helper(encoder_helper, df, category_lst, response):
    '''
    test encoder helper
    '''
    try:
        result_df = encoder_helper(df, category_lst, response)
        assert result_df is not None
        logging.info("SUCCESS: Encoding step passes!")
        return result_df
    except AssertionError as err:
        logging.error("ERROR: Encoding step failed! Encoding returns NONE!")
        raise err

def test_perform_feature_engineering(perform_feature_engineering, X, predictive_var):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(X, predictive_var)
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        logging.info("SUCCESS: perform_feature_engineering passes!")
        return X_train, X_test, y_train, y_test
    except AssertionError as err:
        logging.error("ERROR: perform_feature_engineering failed! return value is None!")
        raise err

def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        path_to_model = "./models"
        models_dir = os.listdir(path_to_model)
        assert len(models_dir)
        logging.info("SUCCESS: train_model passes. Models are saved to local!")
    except AssertionError as err:
        logging.error("ERROR: train_model failed! No model is saved to local!")
        raise err

if __name__ == "__main__":
    # prepare variables to test
    CATEGORY_LST = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    RESPONSE = 'Churn'

    # test each function
    DATA = test_import(cls.import_data)
    test_eda(cls.perform_eda, DATA)
    X_df = test_encoder_helper(cls.encoder_helper, DATA, CATEGORY_LST, RESPONSE)
    predictive_var = DATA[RESPONSE]
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, X_df, predictive_var)
    test_train_models(cls.train_models, X_train, X_test, y_train, y_test)
