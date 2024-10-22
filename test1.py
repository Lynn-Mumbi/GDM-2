import pandas as pd



GDM7th=pd.read_csv("C:\\Users\\lenovo\\PycharmProjects\\pythonProject\\Datasets\\GDM Dataset 7th Sep.csv")
GDM7th = GDM7th[GDM7th['Sex']== "Female"]
print(GDM7th.shape)
print(GDM7th.columns)
print(GDM7th[GDM7th['Gestational Diabetes'] == "Yes"])
