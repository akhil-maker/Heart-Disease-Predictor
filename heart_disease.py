import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

heart_data = 'Heart Disease Predictor/heart.csv' #get data

data = pd.read_csv(heart_data) #read csv using pandas
# data = data.drop(columns=['oldpeak', 'slope', 'ca', 'thal', 'fbs', 'restecg', 'exang'])

#renaming some of features
data = data.rename(columns={'age':'age', 'sex':'gender', 'cp':'chest pain', 'trestbps':'blood pressure', 'chol':'cholestrol level', 'thalach':'max heart rate'})

#print sample datasets
print(data.sample(5))

#print description of data
print(data.describe())

#extract target and data without target
predictors = data.drop("target", axis=1)
target = data["target"]

#Train test data split
xTrain, xTest, yTrain, yTest = train_test_split(predictors, target, test_size=0.20, random_state=0)

#build model for logistic regression
model = linear_model.LogisticRegression(solver='liblinear')

#fit the data into the model
model.fit(xTrain, yTrain)

#get target for test data
data_y_predict = model.predict(xTest)

#test with given datasets
df = pd.DataFrame({'age':63, 'sex':1, 'cp':3, 'trestbps':145, 'chol':233, 'fbs':1, 'restecg':0, 'thalach':150, 'exang':0, 'oldpeak':2.3, 'slope':0, 'ca':0, 'thal':1}, index=[0])
df1 = pd.DataFrame({'age':67, 'sex':1, 'cp':0, 'trestbps':120, 'chol':229, 'fbs':0, 'restecg':0, 'thalach':129, 'exang':1, 'oldpeak':2.6, 'slope':1, 'ca':2, 'thal':3}, index=[0])
print(df)

#print the predicted target of model
print(model.predict(df1))
result = model.predict(df)
print(result)
a = np.array(1)
if result.astype('int') == a.astype('int'):
    msg = "Suffering from heart disease"
else:
    msg = "You are not suffering from heart disease"
print(msg)

#check the accuracy of method
score = round(accuracy_score(data_y_predict, yTest)*100, 2)
print(score)