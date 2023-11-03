###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

from scipy.stats import f as fisher_f
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import math


print("Question 1:")


df = pd.read_csv("cmg_with_color.csv")
model = LinearRegression(fit_intercept=True)

file_length = len(df.index)
i = 0
current_month = -1

months_w_change = 0
year1_changes = 0
year2_changes = 0
while i < file_length:
	if df['Year'].iloc[i].astype(str) == "2019":
		i = file_length
		continue

	current_month = df['Month'].iloc[i]
	j = i
	while df['Month'].iloc[j] == current_month:
		j += 1
	
	month_length = j - i

	left_side = 2
	sig_month = False
	while left_side <= month_length - 2:
		right_side = month_length - left_side

		x_left = np.arange(1, left_side + 1)
		x_left = x_left[:, np.newaxis]
		x_right = np.arange(1, right_side + 1)
		x_right = x_right[:, np.newaxis]

		y_left = df['Adj Close'][(df.index >= i) & (df.index < i + left_side)]
		y_right = df['Adj Close'][(df.index >= i + left_side) & (df.index < j)]

		model.fit(x_left, y_left)
		L1 = model.score(x_left, y_left) # left score
		model.fit(x_right, y_right)
		L2 = model.score(x_right, y_right) # right score
		n = month_length

		
		p_value1 = fisher_f.cdf(L1, 2, n - 4)
		p_value2 = fisher_f.cdf(L2, 2, n - 4)

		if abs(p_value1 - p_value2) > 0.1:
			if not sig_month:
				print("Month", current_month, "of year", df['Year'].iloc[i], "has significant price changes")
				sig_month = True
				months_w_change += 1
			if df['Year'].iloc[i].astype(str) == "2017":
				year1_changes += 1
			elif df['Year'].iloc[i].astype(str) == "2018":
				year2_changes += 1


		left_side += 1

	i = j


print("\nQuestion 2:\n" + str(months_w_change) + " months exhibit significant price changes")

print("\nQuestion 3:")
if year1_changes > year2_changes:
	print("There were more significant changes in year 1 (" + str(year1_changes)
		+ ") than in year 2 (" + str(year2_changes) + ")")
elif year2_changes > year1_changes:
	print("There were less significant changes in year 1 (" + str(year1_changes)
		+ ") than in year 2 (" + str(year2_changes) + ")")
else:
	print("Year 1 and year 2 had equally as many significant changes ("
		+ str(year2_changes) + ")")




