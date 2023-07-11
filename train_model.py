import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ml_tools as tools

LEARNING_RATE = 0.35
EPOCHS = 200

def display_plot(headers,mileage, price, estimate):
	plt.figure(figsize=(10,6))
	plt.xlabel(headers[0])
	plt.ylabel(headers[1])
	plt.scatter(mileage, price, marker='+', label = 'data')
	plt.plot(mileage, estimate, color='red', label = 'estimate')
	plt.show()

def read_csv(file_path):
	headers = np.loadtxt(file_path, delimiter=',', unpack=True, max_rows=1, dtype=str)
	mileage,price= np.loadtxt(file_path, delimiter=',', unpack=True, skiprows=1, dtype=int)
	return headers,mileage,price

def estimate_price(theta_slope, theta_intercept, mileage):
	return theta_slope * mileage + theta_intercept

def cost_function(theta_slope, theta_intercept, mileage, price, m_length):
	cost = 0
	for i in range(m_length):
		f = estimate_price(theta_slope, theta_intercept, mileage[i])
		cost = cost + (f - price[i]) ** 2
	return cost / (m_length * 2)

def derivated_slope(theta_slope, theta_intercept, mileage, price, m_length):
	sum_ = 0
	for mileage_i, price_i in zip(mileage, price):
		sum_ = sum_ + (estimate_price(theta_slope, theta_intercept, mileage_i) - price_i) * mileage_i
	return sum_ / m_length

def derivated_fix(theta_slope, theta_intercept, mileage, price, m_length):
	sum_ = 0
	for mileage_i, price_i in zip(mileage, price):
		sum_ = sum_ + estimate_price(theta_slope, theta_intercept, mileage_i) - price_i
	return sum_ / m_length
			
def linear_regression(mileage, price):
	m_length = mileage.shape[0]
	cost_array = []
	theta_slope = 0
	theta_intercept = 0
	for _ in range(EPOCHS):
		cost = cost_function(theta_slope, theta_intercept, mileage, price, m_length)
		cost_array.append(cost)
		d_slope = derivated_slope(theta_slope, theta_intercept, mileage, price, m_length)
		d_fix = derivated_fix(theta_slope, theta_intercept, mileage, price, m_length)
		theta_slope = theta_slope - LEARNING_RATE * d_slope
		theta_intercept = theta_intercept - LEARNING_RATE * d_fix
		if (len(cost_array) > 1 and (cost_array[-2] < cost_array[-1] or cost_array[-2] - cost_array[-1] < 10e-9)):
			print("Cout stabilisé à l'epoch", _)
			break
	return theta_slope, theta_intercept

def train_model(file_path):
	headers,mileage,price = read_csv(file_path)
	normalized_mileage = tools.normalize_array(mileage)
	normalized_price = tools.normalize_array(price)
	theta_slope,theta_intercept = linear_regression(normalized_mileage,normalized_price)
	display_plot(headers,normalized_mileage,normalized_price, estimate_price(theta_slope, theta_intercept, normalized_mileage))
	
if __name__ == '__main__':
	if (len(sys.argv) <= 1):
		sys.exit('Usage: python3 train.py <csv file>')
	try:
		tools.is_valid_path(sys.argv[1])
	except Exception as e:
		sys.exit(e)
	train_model(sys.argv[1])
