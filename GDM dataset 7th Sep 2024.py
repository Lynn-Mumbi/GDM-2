import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score

import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


GDM7th=pd.read_csv("C:\\Users\\lenovo\\PycharmProjects\\pythonProject\\Datasets\\GDM Dataset 7th Sep.csv")

GDM7th = GDM7th[GDM7th['Sex']== "Female"]
GDM7th = GDM7th.drop('Glucoselevelblood', axis=1)
print(GDM7th.shape)
#print(GDM7th.columns)
#print(GDM7th.isna().sum())

#print(GDM7th.info())

# changing data types
#removing non-numeric data types from FBG and PostPrandium and replacing with nan
GDM7th['Glucoselevel0minblood']=pd.to_numeric(GDM7th['Glucoselevel0minblood'], errors='coerce')
# drop the nan values
GDM7th= GDM7th.dropna(subset=['Glucoselevel0minblood'])
print(GDM7th.info())
#print(GDM7th['Glucoselevel0minblood'].unique())
###########################################################################
GDM7th['Glucoselevel120minblood']=pd.to_numeric(GDM7th['Glucoselevel120minblood'], errors='coerce')
# drop the nan values
GDM7th= GDM7th.dropna(subset=['Glucoselevel120minblood'])
print(GDM7th.info())
#print(GDM7th['Glucoselevel120minblood'].unique())

# the Gravida column has 7 rows with missing data we'll drop that
GDM7th = GDM7th.dropna(subset =['Gravida'])

#missing values any?
print(GDM7th.isna().sum())

#plotting distribution of columns with missing values
'''fig, ax = plt.subplots(1,2)

ax[0].hist(GDM7th['SystolicBloodPressureCuff'], bins=20)
ax[1].hist(GDM7th['Diastolic Blood Pressure'],bins=20)

ax[0].set_title("SystolicBloodPressureCuff with missing values")
ax[1].set_title("Diastolic Blood Pressure with missing values")

plt.show()
'''
# imputing the missing values
# SBP impute
mean_Sys = GDM7th['SystolicBloodPressureCuff'].mean()
GDM7th['SystolicBloodPressureCuff'] = GDM7th['SystolicBloodPressureCuff'].fillna(mean_Sys)

mean_dys = GDM7th['Diastolic Blood Pressure'].mean()
GDM7th['Diastolic Blood Pressure'] = GDM7th['Diastolic Blood Pressure'].fillna(mean_dys)

# plotting the distribution
'''fig, ax = plt.subplots(1,2)

ax[0].hist(GDM7th['SystolicBloodPressureCuff'], bins=20)
ax[1].hist(GDM7th['Diastolic Blood Pressure'],bins=20)

ax[0].set_title("SystolicBloodPressureCuff with missing values")
ax[1].set_title("Diastolic Blood Pressure with missing values")

plt.show()
'''
pd.set_option('future.no_silent_downcasting', True)

print(GDM7th.isna().sum()) # Great no missing values by this time

# turning Obese values to numeric (1 for yes and 0 for no)
GDM7th['Obese?']= GDM7th['Obese?'].replace({'Yes':1,'No':0}).astype(int)
GDM7th['Obese?'].astype('int')
#print(GDM7th['Obese?'].unique())

# turning Gestational Diabetes values to numeric (1 for yes an 0 for no)
GDM7th['Gestational Diabetes']= GDM7th['Gestational Diabetes'].replace({'Yes':1,'No':0}).astype(int)
GDM7th['Gestational Diabetes'].astype('int')


# turning Sex to numeric value 1
GDM7th['Sex']= GDM7th['Sex'].replace('Female', 1)
GDM7th['Sex'].astype('int')

# confirm the data types
print(GDM7th.info()) # all are numeric
'''print(GDM7th['Gestational Diabetes'].unique())'''
print(GDM7th['Sex'].unique())

print(GDM7th.columns)

# correlation analysis
cols_to_corr= [
       'AgeAtStartOfSpell','Body Mass Index at Booking', 'Obese?', 'Parity', 'Gravida',
       'Glucoselevel0minblood', 'Glucoselevel120minblood', 'SystolicBloodPressureCuff',
       'Diastolic Blood Pressure', 'Gestation', 'No_Of_previous_Csections'
]

# using spearman correlation for individual columns
correlations = {col: GDM7th[col].corr(GDM7th['Gestational Diabetes'], method='spearman') for col in cols_to_corr}

# creating df
corr_df = pd.DataFrame(list(correlations.items()), columns=['Column','Spearman Correlation'])
print(corr_df.sort_values(by = 'Spearman Correlation', ascending= False))

# checking for outliers
all_columns=GDM7th[['Body Mass Index at Booking','Glucoselevel0minblood',
                    'Glucoselevel120minblood',
                    'SystolicBloodPressureCuff','Diastolic Blood Pressure', 'Gestation',
                    'No_Of_previous_Csections']]
all_columns.plot(kind= "box", subplots = True , layout =(1,9), sharey = False, sharex= False)
plt.suptitle("Box plots for columns with outliers")
#plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.show()

#counting outliers in a column
def count_outliers(column):
    Q1 = GDM7th[column].quantile(0.25)
    Q3 = GDM7th[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    return GDM7th[(GDM7th[column] < lower_bound) | (GDM7th[column] > upper_bound)].shape[0]

columns_with_outliers = ['AgeAtStartOfSpell','Body Mass Index at Booking',
       'Glucoselevel0minblood', 'Glucoselevel120minblood', 'SystolicBloodPressureCuff',
       'Diastolic Blood Pressure', 'Gestation', 'No_Of_previous_Csections'
]

# Calculate outliers count for each column
outliers_count = {col: count_outliers(col) for col in columns_with_outliers}

print(outliers_count)
print(GDM7th.shape)

#removing outliers by imputation
def impute_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median = df[col].median()
        df[col] = df[col].apply(lambda x: median if x > upper_bound or x < lower_bound else x)
    return df

gdm_imputed = impute_outliers(GDM7th, columns_with_outliers)

#splitting data into features and the target variable
X = GDM7th.drop(['Gestational Diabetes', 'UID'], axis=1)
y = GDM7th['Gestational Diabetes']

#spliting the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 1999)

# standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled features back to a DataFrame
X_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
print(X_scaled_df)

# Stochastic Gradient Boosting
sgbt_classifier = GradientBoostingRegressor(max_depth=1, subsample=0.8, max_features= 0.2, n_estimators=300, random_state= 1999)
sgbt_classifier.fit(X_train_scaled,y_train)
#predict
y_pred = sgbt_classifier.predict(X_test_scaled)

# evaluating  the model
sgbt_mse = MSE(y_test,y_pred)
print(f'SGBT Mean Squared Error: {sgbt_mse}')
#RMSE test
sgbt_rmse = sgbt_mse ** (1/2)
print('SGBT RMSE: {:.2f}'.format(sgbt_rmse))
# calculating the accuracy score
'''sgbt_accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy: {sgbt_accuracy}')'''






