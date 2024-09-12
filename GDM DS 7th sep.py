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


GDM7th=pd.read_csv("C:\\Users\\USER\\PycharmProjects\\pythonProject\\Datasets\\GDM Dataset 7th Sep.csv")

GDM7th = GDM7th[GDM7th['Sex']== "Female"]
GDM7th = GDM7th.drop('Glucoselevelblood', axis=1)
print(GDM7th.shape)
#print(GDM7th.columns)
#print(GDM7th.isna().sum())



#print(GDM7th.info())

# changing data types
#removing non numeric data types from FBG and PostPrandium and replacing with nan
GDM7th['Glucoselevel0minblood']=pd.to_numeric(GDM7th['Glucoselevel0minblood'], errors='coerce')
# drop the nan values
GDM7th= GDM7th.dropna(subset=['Glucoselevel0minblood'])
print(GDM7th.info())
print(GDM7th['Glucoselevel0minblood'].unique())
###########################################################################
GDM7th['Glucoselevel120minblood']=pd.to_numeric(GDM7th['Glucoselevel120minblood'], errors='coerce')
# drop the nan values
GDM7th= GDM7th.dropna(subset=['Glucoselevel120minblood'])
print(GDM7th.info())
print(GDM7th['Glucoselevel120minblood'].unique())


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
GDM7th['SystolicBloodPressureCuff'].fillna(mean_Sys,inplace=True)

mean_dys = GDM7th['Diastolic Blood Pressure'].mean()
GDM7th['Diastolic Blood Pressure'].fillna(mean_dys,inplace=True)

# plotting the distribution
'''fig, ax = plt.subplots(1,2)

ax[0].hist(GDM7th['SystolicBloodPressureCuff'], bins=20)
ax[1].hist(GDM7th['Diastolic Blood Pressure'],bins=20)

ax[0].set_title("SystolicBloodPressureCuff with missing values")
ax[1].set_title("Diastolic Blood Pressure with missing values")

plt.show()'''

print(GDM7th.isna().sum()) # Great no missing values by this time

# turning Obese values to numeric (1 for yes and 0 for no)
GDM7th['Obese?']= GDM7th['Obese?'].replace({'Yes':1,'No':0})
GDM7th['Obese?'].astype('int')
#print(GDM7th['Obese?'].unique())

# turning Gestational Diabetes values to numeric (1 for yes an 0 for no)
GDM7th['Gestational Diabetes']= GDM7th['Gestational Diabetes'].replace({'Yes':1,'No':0})
GDM7th['Gestational Diabetes'].astype('int')

# turning Sex to numeric value 1
GDM7th['Sex']= GDM7th['Sex'].replace('Female', 1)
GDM7th['Sex'].astype('int')

# confirm the data types
print(GDM7th.info()) # all are numeric
'''print(GDM7th['Gestational Diabetes'].unique())
print(GDM7th['Sex'].unique())'''

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

# Dash app initialization
app = Dash(__name__)

app.layout = html.Div([
    html.H1("GDM Risk Assessment Dashboard"),

    # Dropdown to select the column
    html.Label("Select a column to view box plot:"),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in all_columns],
        value=all_columns[0],  # Default selection
        clearable=False
    ),

    # Boxplot for the selected column
    dcc.Graph(id='boxplot-graph')
])


# Callback to update the box plot based on dropdown selection
@app.callback(
    Output('boxplot-graph', 'figure'),
    [Input('column-dropdown', 'value')]
)
def update_boxplot(selected_column):
    # Create the box plot for the selected column
    boxplot = go.Figure(
        data=[go.Box(y=GDM7th[selected_column], name=selected_column)],
        layout=go.Layout(
            title=f"Box plot for {selected_column}",
            yaxis_title='Value',
            xaxis_title='Feature'
        )
    )
    return boxplot


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

