# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: POOJA SRI P
RegisterNumber: 212224230197
*/

import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
data.head()

![image](https://github.com/user-attachments/assets/d62e6005-beae-4418-96af-6a9cf64acf7c)

data.info()

![image](https://github.com/user-attachments/assets/3f7372a2-2b9c-4b2c-84fd-ab99ab200c66)

data.isnull().sum()

![image](https://github.com/user-attachments/assets/c5206f93-13ef-4a45-91f4-d1c1518623c0)

data.head() for salary

![image](https://github.com/user-attachments/assets/58955dc2-b719-47a6-b2ce-1f3df4dccec5)

MSE value

![image](https://github.com/user-attachments/assets/3965091e-8d61-4222-b350-ff925ad51ef2)


r2 value

![image](https://github.com/user-attachments/assets/3f5d3276-815c-421c-b65c-9a70fc84f646)

data prediction

![image](https://github.com/user-attachments/assets/6fd3424c-7c9d-468f-9e65-604aa30a94ce)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
