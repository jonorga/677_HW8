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


def Q1EvaluateW(w_row):
	w = w_row['W']
	i = w
	model = LinearRegression(fit_intercept=True)
	trade_count = 0
	trade_sum = 0
	long_position = False
	short_position = False
	cur_stock = 0
	last_i = 0
	while i < file_length:
		if df['Year'].iloc[i].astype(str) == "2018":
			last_i = i
			i = file_length
			continue

		x = np.arange(1, w + 1)
		y = df['Adj Close'][(df.index <= i) & (df.index > i - w)]
		x_2 = x[:, np.newaxis]
		model.fit(x_2, y)

		tmr_prediction = model.predict([[i + 1]])
		today_true = df['Adj Close'].iloc[i]

		if tmr_prediction > today_true:
			if not long_position:
				cur_stock = 100 / today_true
				long_position = True
			if short_position:
				trade_sum += 100 - (cur_stock * today_true)
				trade_count += 1
				short_position = False
		elif tmr_prediction < today_true:
			if long_position:
				trade_sum += (cur_stock * today_true) - 100
				trade_count += 1
				long_position = False
			if not short_position:
				cur_stock = 100 / today_true
				short_position = True
		i += 1

	if short_position:
		trade_sum += 100 - (cur_stock * df['Adj Close'].iloc[last_i])
		trade_count += 1
	if long_position:
		trade_sum += (cur_stock * df['Adj Close'].iloc[last_i]) - 100
		trade_count += 1

	if trade_count == 0:
		return -1
	else:
		return trade_sum / trade_count

q1_df = pd.DataFrame(np.arange(5, 31), columns=['W'])
q1_df['Average P/L'] = q1_df.apply(Q1EvaluateW, axis=1)



