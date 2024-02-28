import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv("train.csv")
Target_column = data.columns[6]
X = np.array(data.drop([Target_column],axis=1))
y = np.array(data[Target_column])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.8)

model = linear_model.LinearRegression()
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)
predict = model.predict(x_test)

for i in range(0,5):
    print(predict[i],x_test[i],y_test[i])
