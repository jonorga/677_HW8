###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
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

def Q1GenerateGraph(data, col1, col2, xlab, ylab, _title, q):
	fig, ax = plt.subplots()
	ax.plot(data[col1], data[col2])
	ax.set(xlabel=xlab, ylabel=ylab,
	       title=_title)
	ax.grid()
	print("Saving " + _title + " graph...")
	fig.savefig("Q" + q + ".png")

Q1GenerateGraph(q1_df, "W", "Average P/L", 'W Value', 'W Average P/L', 'W Average P/L by trade', "1_W_averagePL")
print("The optimal W* value of W is 8")


print("\n")
# Question 2 ====================================================================================================
print("Question 2:")


def Q2GenerateDF():
	data = []

	i = 0
	count = 1
	model = LinearRegression(fit_intercept=True)
	while i < file_length:
		if df['Year'].iloc[i].astype(str) == "2017":
			i += 1
			continue
		if df['Year'].iloc[i].astype(str) == "2019":
			i = file_length
			continue

		x = np.arange(1, 9)
		y = df['Adj Close'][(df.index <= i) & (df.index > i - 8)]
		x_2 = x[:, np.newaxis]
		model.fit(x_2, y)

		data.append([count, model.score(x_2, y)])
		count += 1

		i += 1
	temp_df = pd.DataFrame(data, columns=['Date_num', 'R2_Val'])
	return temp_df

q2_df = Q2GenerateDF()
Q1GenerateGraph(q2_df, "Date_num", "R2_Val", 'Day', 'R Squared Value', 'R Squared value by Day for Y2', "2_R2byDay")



