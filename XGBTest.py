import pandas as pd
import numpy as np

heart_failure = pd.read_csv("heart_failure_clinical_records_dataset.csv")

heart_failure.dtypes

heart_failure.describe()

list_lower_out = []
list_higher_out = []
heart_failure_descriptive_stats = heart_failure.describe()
for column in list(heart_failure_descriptive_stats.columns):
    IQR = heart_failure_descriptive_stats[column][6] - heart_failure_descriptive_stats[column][4]
    
    lower_boundary_for_outliers = heart_failure_descriptive_stats[column][4] - (IQR * 1.5)
    higher_boundary_for_outliers = heart_failure_descriptive_stats[column][6] + (IQR * 1.5)
    
    for value in heart_failure[column]:
        if value < lower_boundary_for_outliers:
            list_lower_out.append(value) 
        if value > higher_boundary_for_outliers:
            list_higher_out.append(value)
    print(column, ':')
    print('lower:', len(list_lower_out))
    print('higher:', len(list_higher_out))
    
##The outliers are relevant to describe specific scenarios for the pacients. Either respresenting 'certain'
##survival or 'certain' death

heart_failure.isna().sum()

heart_failure.corr()

##There is no need to drop any column because there isn't high collinearity between any of them

##Create engine to connect with SQL and create a table in a database

from sqlalchemy import create_engine

import pymysql

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",
                               pw="Hisbenfica97",
                               db="final_project"))

heart_failure.to_sql("heart_failure", con = engine, if_exists = "append")

##Initialize a connector to query the database that is on SQL

import mysql.connector

pass_er = 'Hisbenfica97'
cnx = mysql.connector.connect(user='root', password=pass_er,
                              host='localhost',
                              database='final_project',
                              auth_plugin='mysql_native_password')

mycursor = cnx.cursor()

mycursor.execute("SELECT age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time FROM final_project.heart_failure") 
x = mycursor.fetchall()

x = pd.DataFrame(x)

mycursor.execute("SELECT DEATH_EVENT FROM final_project.heart_failure")
y = mycursor.fetchall()

y = pd.DataFrame(y)

##Split the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y)

##Grid Search to get the best parameters for the XGBoosting

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

max_depth = [4,5,6,7,8,9,10]
min_child_weight = [1,2,3,4,5,6]
gamma = [0.1, 0.2, 0.3, 0.4]
learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
subsample = [0.5, 0.6, 0.7, 0.8]

grid = {'booster': ['gbtree']
    ,'max_depth': max_depth
    ,'min_child_weight': min_child_weight
    ,'gamma': gamma
    ,'learning_rate': learning_rate
    ,'subsample': subsample
    ,'objective': ['binary:hinge']
    ,'verbosity': [3]
}

booster = xgb.XGBClassifier('binary:hinge')

grid_search = GridSearchCV(estimator = booster, param_grid = grid,  cv = 3)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
##{'booster': 'gbtree', 'gamma': 0.1, 'learning_rate': 0.2, 'max_depth': 5, 'min_child_weight': 4, 'objective': 'binary:hinge', 'subsample': 0.7, 'verbosity': 3}
