import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
data = pd.read_csv("Salary_dataset.csv", sep=",") #Get Csv file and sep by Koma
columns  = data.columns #Get column names
X = np.array(data.drop([columns[0]], axis=1))  #Get x data which is  Features
y = np.array(data[columns[0]]) #Get y data which is we want predict

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size= 0.1) #train test data split
model = linear_model.LinearRegression() # model
model.fit(x_train,y_train) # train model
acc = model.score(x_test,y_test) # accuracy
print(acc)
#print('Coefficient: \n', model.coef_) # These are each slope value
#print('Intercept: \n', model.intercept_) # This is the intercept

predic = model.predict(x_test) #test our model with test data 

#print(mean_squared_error(y_test, predic)) #mean squared error

# show the prediction, which data model get and real answer
#for i in range(len(predic)):
#    print(predic[i], x_test[i], y_test[i])
