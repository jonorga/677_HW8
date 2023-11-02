###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
		w_total = 0
		w_correct = 0
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
			w_total += 1
			if today_true < tmr_prediction and df['Color'].iloc[i + 1] == "Green":
				w_correct += 1
			elif today_true  > tmr_prediction and df['Color'].iloc[i + 1] == "Red":
				w_correct += 1
			i += 1
		w_acc = -1 if w_total == 0 else (w_correct / w_total)
		acc_list.append(w_acc)
		w += 1
	return acc_list


def Q1GenerateGraph(data1, data2, data3):
	fig, ax = plt.subplots()
	ax.plot(data1['W'], data1['Accuracy'], label="1st Degree")
	ax.plot(data2['W'], data2['Accuracy'], label="2nd Degree")
	ax.plot(data3['W'], data3['Accuracy'], label="3rd Degree")
	ax.set(xlabel='W Value', ylabel='Accuracy',
	       title='W Accuracy by Value')
	fig.legend(loc="upper right")
	ax.grid()
	print("Saving Q1 graph...")
	fig.savefig("Q1_WAccuracy_Graph.png")


deg1_acc = Q1GenerateData(1)
deg2_acc = Q1GenerateData(2)
deg3_acc = Q1GenerateData(3)

deg1_data = []
deg2_data = []
deg3_data = []
w = 5
while w <= 12:
	deg1_data.append([w, deg1_acc[w - 5]])
	deg2_data.append([w, deg2_acc[w - 5]])
	deg3_data.append([w, deg3_acc[w - 5]])
	w += 1

deg1_df = pd.DataFrame(deg1_data, columns=['W', 'Accuracy'])
deg2_df = pd.DataFrame(deg2_data, columns=['W', 'Accuracy'])
deg3_df = pd.DataFrame(deg3_data, columns=['W', 'Accuracy'])

Q1GenerateGraph(deg1_df, deg2_df, deg3_df)

print("\n")
# Question 2 ====================================================================================================
print("Question 2:")


def Q2ComputeAccuracy(deg, d_str, w):
	i = w + 50
	w_total = 0
	w_correct = 0
	predicted = []
	actual = []
	while i < 100:
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
		w_total += 1
		actual.append(df['Color'].iloc[i + 1])
		if today_true < tmr_prediction:
			predicted.append("Green")
			if df['Color'].iloc[i + 1] == "Green":
				w_correct += 1
		elif today_true  > tmr_prediction:
			predicted.append("Red")
			if df['Color'].iloc[i + 1] == "Red":
				w_correct += 1
		i += 1
	w_acc = -1 if w_total == 0 else (w_correct / w_total)
	print("For year 2, using a " + d_str + " degree polynomial function with a W value of " + str(w)
		+ " resulted in an accuracy of " + str(round(w_acc * 100, 2)) + "%")
	return [actual, predicted]


deg1_ct_vals = Q2ComputeAccuracy(1, "1st", deg1_df['W'].loc[deg1_df['Accuracy'].idxmax()])
deg2_ct_vals = Q2ComputeAccuracy(2, "2nd", deg2_df['W'].loc[deg2_df['Accuracy'].idxmax()])
deg3_ct_vals = Q2ComputeAccuracy(3, "3rd", deg3_df['W'].loc[deg3_df['Accuracy'].idxmax()])


print("\n")
# Question 3 ====================================================================================================
print("Question 3:")

y_actu1 = pd.Series(deg1_ct_vals[0], name='Actual')
y_pred1 = pd.Series(deg1_ct_vals[1], name='Predicted')
y_actu2 = pd.Series(deg2_ct_vals[0], name='Actual')
y_pred2 = pd.Series(deg2_ct_vals[1], name='Predicted')
y_actu3 = pd.Series(deg3_ct_vals[0], name='Actual')
y_pred3 = pd.Series(deg3_ct_vals[1], name='Predicted')

deg1_cm = pd.crosstab(y_actu1, y_pred1)
deg2_cm = pd.crosstab(y_actu2, y_pred2)
deg3_cm = pd.crosstab(y_actu3, y_pred3)

print("1st degree polynomial confusion matrix:")
print(deg1_cm)
print("\n2nd degree polynomial confusion matrix:")
print(deg2_cm)
print("\n3rd degree polynomial confusion matrix:")
print(deg3_cm)


print("\n")
# Question 4 ====================================================================================================
print("Question 4:")


def LinearRegTrade(y_actu, y_pred, w):
	balance = 100
	offset = w + 50
	i = offset
	while i < 100:
		today_stock = balance / df['Close'].iloc[i]
		tmr_stock = balance / df['Close'].iloc[i + 1]
		difference = abs(today_stock - tmr_stock)
		if y_actu.iloc[i - offset] == y_pred.iloc[i - offset]:
			balance += difference * df["Close"].iloc[i + 1]
		else:
			balance -= difference * df["Close"].iloc[i + 1]
		i += 1
	return round(balance, 2)


deg1_result = LinearRegTrade(y_actu1, y_pred1, deg1_df['W'].loc[deg1_df['Accuracy'].idxmax()])
deg2_result = LinearRegTrade(y_actu2, y_pred2, deg2_df['W'].loc[deg2_df['Accuracy'].idxmax()])
deg3_result = LinearRegTrade(y_actu3, y_pred3, deg3_df['W'].loc[deg3_df['Accuracy'].idxmax()])

print("Starting with $100, trading results by polynomial degree and W*:")

print("1st degree: W* = " + str(deg1_df['W'].loc[deg1_df['Accuracy'].idxmax()]) + ", final balance = $"
	+ str(deg1_result))
print("2nd degree: W* = " + str(deg2_df['W'].loc[deg2_df['Accuracy'].idxmax()]) + ", final balance = $"
	+ str(deg2_result))
print("3rd degree: W* = " + str(deg3_df['W'].loc[deg3_df['Accuracy'].idxmax()]) + ", final balance = $"
	+ str(deg3_result))



