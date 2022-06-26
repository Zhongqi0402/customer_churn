# Predict Customer Churn

## Project Description
In this project, I used kaggle dataset to identify credit card customers who are likely to churn.
I performed end-to-end machine learning pipeline, from importing data, EDA, feature engineering,
training and testing models. I made sure the code has high standard documentation in all levels,
as well as proper testing and logging to track process. This pipeline would meet production level.

## Files and data description
The file structure is as follows.  
.root  
├── churn_notebook.ipynb # Contains the code to be refactored  
├── churn_library.py     # Define the functions  
├── churn_script_logging_and_tests.py # Finish tests and logs  
├── README.md            # Provides project overview, and instructions to use the code  
├── data                 # Read this data  
│   └── bank_data.csv  
├── images               # Store EDA results  
│   ├── eda  
│   └── results  
├── logs                 # Store logs  
└── models               # Store models  

## Running Files
In order to run the scripts, make sure you are at root folder of this project.
run `python churn_library.py` to run the pipeline and run `python churn_script_logging_and_tesing.py`
to run the logging and testing script.



