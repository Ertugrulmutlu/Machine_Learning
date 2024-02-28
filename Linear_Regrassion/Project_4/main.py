import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing

data = pd.read_csv("insurance.csv")
Column = data.columns
le = preprocessing.LabelEncoder()
age = np.array(data[Column[0]])
sex = le.fit_transform(list(data[Column[1]]))
bmi = np.array(data[Column[2]])
children = np.array(data[Column[3]])
smoker = le.fit_transform(list(data[Column[4]]))
region =le.fit_transform(list(data[Column[5]]))
charges = np.array(data[Column[6]]) # our target
X = list(zip(age,sex,bmi,children,smoker,region))
y = list(charges)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
model = linear_model.LinearRegression()
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
#print(acc)
predict = model.predict(x_test)

for i in range(len(predict)):
    print(predict[i],x_test[i],y_test[i])
