import sys
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import ml_tools as tools
import argparse

LEARNING_RATE = 0.35
EPOCHS = 500

def display_plot(headers,mileage, price, estimate):
	fig = px.scatter(x=mileage, y=price, title='Price / Mileage')
	fig.add_scatter(x=mileage, y=estimate, mode='lines', name='estimate')
	fig.show()

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
	normalized_theta_slope,normalized_theta_intercept = linear_regression(normalized_mileage,normalized_price)
	normalized_estimated_price = estimate_price(normalized_theta_slope, normalized_theta_intercept, normalized_mileage)
	display_plot(headers, normalized_mileage, normalized_price, normalized_estimated_price)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a linear regression model')
	parser.add_argument('file_path', metavar='file_path', type=str, help='csv file path')
	args = parser.parse_args()
	try:
		tools.is_valid_path(args.file_path)
	except Exception as e:
		sys.exit(e)
	train_model(args.file_path)
