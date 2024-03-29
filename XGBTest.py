import pandas as pd

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
    
##The outliers are relevant to describe specific scenarios for the pacients. Either respresenting
##'certain' survival or 'certain' death

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

heart_failure.to_sql("heart_failure", con = engine, if_exists = "fail")

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
x.columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
             'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
             'smoking', 'time']

mycursor.execute("SELECT DEATH_EVENT FROM final_project.heart_failure")
y = mycursor.fetchall()

y = pd.DataFrame(y)
y.columns = ['DEATH_EVENT']

##Split the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y)

##Now let's oversample the data to deal with class imbalance
from sklearn.utils import resample

train = pd.concat([X_train, y_train],axis=1)

train.head()

no_death  = train[train['DEATH_EVENT']==0]
yes_death = train[train['DEATH_EVENT']==1]

yes_death_oversampled = resample(yes_death, replace=True, n_samples = len(no_death), random_state=0)

train_oversampled = pd.concat([no_death, yes_death_oversampled])

y_train_over = train_oversampled['DEATH_EVENT'].copy()
X_train_over = train_oversampled.drop('DEATH_EVENT',axis = 1).copy()

##Grid Search to get the best parameters for the XGBoosting

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

max_depth = [4,5,6,7,8,9,10]
min_child_weight = [1,2,3,4,5,6]
gamma = [0.1, 0.2, 0.3, 0.4]
learning_rate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
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

booster = xgb.XGBClassifier(grid)

grid_search = GridSearchCV(estimator = booster, param_grid = grid,  cv = 3)

grid_search.fit(X_train_over, y_train_over)

print(grid_search.best_params_)
print(grid_search.best_score_)

##Create the model in order to calculate accuracy and check the number of false negatives

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

##for i in range(100):
    
param = grid_search.best_params_
    
    
d_train = xgb.DMatrix(X_train_over, y_train_over)
d_test = xgb.DMatrix(X_test, y_test)
    
model = xgb.train(param, d_train, num_boost_round = 35)
    
y_pred = model.predict(d_test)
    
print(roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
    
##Auxiliary piece of code used with the for cicle commented in line 131 to determine the best
#parameter num_boost_rouund

'''list1 = []
list2 = []

if roc_auc_score(y_test, y_pred) > 0.80:
    list1.append((roc_auc_score(y_test, y_pred)))
       list1.append(i)
       list2.append(confusion_matrix(y_test, y_pred)[0][1])
       list2.append(i)'''

##KNN to compare results with the XGBoost

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

for i in range(1, 100):
    knn = KNeighborsClassifier(i)
    
    knn.fit(X_train_over, y_train_over)
    
    ##Standardized data
    
    scale = StandardScaler().fit(pd.concat([X_train_over, X_test]))
    
    X_train_std = scale.transform(X_train_over)
    X_test_std  = scale.transform(X_test)
    
    model_knn = knn.fit(X_train_std, y_train_over)
    
    y_pred = model_knn.predict(X_test_std)
    
    if roc_auc_score(y_test, y_pred) > 0.75:
        print(roc_auc_score(y_test, y_pred), i)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        
#Create a file for the model to use with the app

import pickle

# save the model to disk
filename = 'xgboost_heart_failure_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

cnx.commit()
mycursor.close()
cnx.close() 
