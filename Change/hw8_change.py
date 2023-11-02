###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

from scipy.stats import f as fisher_f
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
#p_value = fisher_f.cdf(f_statistics, 2, n - 4)


df = pd.read_csv("cmg_with_color.csv")
model = LinearRegression(fit_intercept=True)

file_length = len(df.index)
i = 0
current_month = -1
while i < file_length:
	if df['Year'].iloc[i].astype(str) == "2019":
		i = file_length
		continue

	current_month = df['Month'].iloc[i]
	j = i
	while df['Month'].iloc[j] == current_month:
		j += 1
	
	# i = first day of the month
	# j = last day of the month
	# do month breakup here
	month_length = j - i

	left_side = 2
	while left_side <= month_length - 2:
		right_side = month_length - left_side

		x_left = np.arange(1, left_side + 1)
		x_left = x_left[:, np.newaxis]
		x_right = np.arange(1, right_side + 1)
		x_right = x_right[:, np.newaxis]

		y_left = df['Adj Close'][(df.index >= i) & (df.index < i + left_side)]
		y_right = df['Adj Close'][(df.index >= i + left_side) & (df.index < j)]

		model.fit(x_left, y_left)
		left_score = model.score(x_left, y_left)

		model.fit(x_right, y_right)
		right_score = model.score(x_right, y_right)
		#p_value = fisher_f.cdf([left_score, right_score]) #, 2, left_side - 4

		if current_month == 1:
			print(p_value)

		left_side += 1

	i = j





