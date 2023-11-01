###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import numpy as np
import pandas as pd

df = pd.read_csv("cmg_weeks.csv")
file_length = len(df.index)

# How this works
if False:
	x = np.arange(1, 6)
	y = df['Close'][(df.index <= 5) & (df.index > 0)].to_numpy()
	z1 = np.polyfit(x, y, 1)
	z2 = np.polyfit(x, y, 2)
	z3 = np.polyfit(x, y, 3)

	print("X & Y:")
	print(x)
	print(y)

	print("Degree 1:")
	print(z1)
	print("Prediction:",(z1[0] * 6) + z1[1])
	# y = mx + b


	print("\nDegree 2:")
	print(z2)
	print("Prediction:", ( z2[0] * (6 ** 2) ) + (z2[1] * 6) + z2[2])
	# y = mx^2 + nx + b

	print("\nDegree 3:")
	print(z3)
	pos = 6
	print("Prediction:", ( z3[0] * (pos ** 3) ) + (z3[1] * (pos ** 2) ) + (z3[2] * pos) + z3[3])
	# y = mx^3 + nx^2 + rx + b


# Question 1 ====================================================================================================
print("Question 1:")

def Q1GenerateData(deg):
	i = 0
	w = 5
	acc_list = []
	while w <= 12:
		i = w
		while i < 50:
			x = np.arange(1, w + 1)
			y = df['Close'][(df.index <= i) & (df.index > i - w)].to_numpy()
			coefs = np.polyfit(x, y, deg)

			next_pos = w + 1
			if deg == 1:
				tmr_prediction = ( coefs[0] * next_pos ) + coefs[1]
			elif deg == 2:
				tmr_prediction = ( coefs[0] * (next_pos ** 2) ) + ( coefs[1] * next_pos ) + coefs[2]
			elif deg == 3:
				tmr_prediction = ( coefs[0] * (next_pos ** 3) ) + ( coefs[0] * (next_pos ** 2) ) 
				tmr_prediction += ( coefs[1] * next_pos ) + coefs[2]

			today_true = df['Close'].iloc[i]
			
			# If tomorrow prediction > today true:
			#	predict green for tomorrow
			# else: red
			# add one to prediction count
			# if prediction is equal to tomorrow true: add one to correct count
			i += 1
		# append to a running array correct / total
		w += 1
	return acc_list

Q1GenerateData(1)