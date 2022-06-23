# library doc string
'''
Author: Matthew Yue
Purpose of file: This file is a library containing all functions to perform
customer churn project.
Date: June 23, 2022
'''

# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    try:
        df = pd.read_csv(pth)
        return df
    except:
        print("Reading in dataframe is not successful.")


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # encode Attrition_Falg column into binary for prediction
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    # categorize different columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
    ]
    quant_columns = [
        'Customer_Age',
        'Dependent_count', 
        'Months_on_book',
        'Total_Relationship_Count', 
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 
        'Credit_Limit', 
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 
        'Total_Amt_Chng_Q4_Q1', 
        'Total_Trans_Amt',
        'Total_Trans_Ct', 
        'Total_Ct_Chng_Q4_Q1', 
        'Avg_Utilization_Ratio'
    ]
    
    # Plot predictive variable
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist()
    plt.savefig("./images/eda/churn_plot.png")
    
    # Plot all quantitative columns
    for col in quant_columns:
        plt.figure(figsize=(20,10)) 
        df[col].hist()
        plt.savefig('./images/eda/'+ col +'.png')
   
    # Plot all categorical variables
    for col in cat_columns:
        plt.figure(figsize=(20,10)) 
        df[col].value_counts('normalize').plot(kind='bar')
        plt.savefig("./images/eda/" + col + ".png")
        
    # Plot correlation heap maps
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig("./images/eda/correlation_heatMap.png")
    
    return
    
    
    

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        current_col_list = []
        current_groups = df.groupby(col).mean()[response]
        
        for val in df[col]:
            current_col_list.append(current_groups.loc[val])
        
        df[col+'_'+response] = current_col_list
        
    result_df = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
                 'Income_Category_Churn', 'Card_Category_Churn']

    result_df[keep_cols] = df[keep_cols]

    return result_df

def perform_feature_engineering(df, predictive_var):
    '''
    input:
              df: pandas dataframe
              predictive_var: the response variable we are interested in 

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X_train, X_test, y_train, y_test = train_test_split(df, predictive_var, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              training predictions and testing predictions from all 
              models built in this function
    '''
    # random forest and logistic regression
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    
    # grid search for best parameters
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    
    # fit random forest and logistic regression
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    
    # predict using both training data and testing data
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


if __name__ == "__main__":
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
    ]
    response = 'Churn'
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    response_var = df[response]
    result_df = encoder_helper(df, category_lst, response)
    X_train, X_test, y_train, y_test = perform_feature_engineering(result_df, response_var)
    y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = train_models(X_train, X_test, y_train, y_test)
    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))