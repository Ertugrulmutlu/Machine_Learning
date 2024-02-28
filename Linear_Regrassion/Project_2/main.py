import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv("cars24-car-price-clean2.csv")#Get Csv file and sep by Koma
columns = data.columns#Get column names
target_column = [columns[3]] # get target column
delete_columns = [columns[[6]],columns[[7]]] #get must delete column
data = data.drop(delete_columns[0], axis=1) # delete
data = data.drop(delete_columns[1], axis=1) #dewlete
X = np.array(data.drop(target_column ,axis=1)) #Get x data which is  Features
y = np.array(data[target_column])#Get y data which is we want predict




x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size= 0.1)#train test data split
print(x_train, x_test, y_train, y_test)
model = linear_model.LinearRegression()# model
model.fit(x_train,y_train)# train model

acc = model.score(x_test,y_test)# accuracy
#print('Coefficient: \n', model.coef_) # These are each slope value
#print('Intercept: \n', model.intercept_) # This is the intercept
print("Accurucyt: ",acc)


predict = model.predict(x_test)
# show the prediction, which data model get and real answer
for i in range(0,5):
    print(predict[i],x_test[i],y_test[i])
    pass