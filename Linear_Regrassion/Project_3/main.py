import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv("wine.csv")
Columns = data.columns
target_column = Columns[1]

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print(X)

