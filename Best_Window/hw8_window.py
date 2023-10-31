###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing
from sklearn import utils

df = pd.read_csv("cmg_with_color.csv")
file_length = len(df.index)

# Question 1 ====================================================================================================
print("Question 1:")


def Q1EvaluateW(w):
	i = w
	model = LinearRegression(fit_intercept=True)
	while i < file_length:
		if df['Year'].iloc[i].astype(str) == "2018":
			i = file_length

		if w == i:		
			#x = df.index[(df.index <= i) & (df.index > i - w)].to_numpy().reshape(-1, 1)
			x = np.arange(1, w + 1)
			y = df['Adj Close'][(df.index <= i) & (df.index > i - w)]
			x_2 = x[:, np.newaxis]
			model.fit(x_2, y)
			#print(y)
			#print(model.predict([[i + 1]]))
			#print(df['Adj Close'].iloc[i + 1])
		i += 1

Q1EvaluateW(5)