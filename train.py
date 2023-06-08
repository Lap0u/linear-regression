import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEARNING_RATE = 0.3
EPOCHS = 30

def displayPlot(headers,mileage, price, estimate):
	plt.figure(figsize=(10,6))
	plt.xlabel(headers[0])
	plt.ylabel(headers[1])
	plt.scatter(mileage, price, marker='+', label = 'data')
	plt.plot(mileage, estimate, color='red', label = 'estimate')
	plt.show()

def readCsv(filePath):
	headers = np.loadtxt(filePath, delimiter=',', unpack=True, max_rows=1, dtype=str)
	mileage,price= np.loadtxt(filePath, delimiter=',', unpack=True, skiprows=1, dtype=int)
	return headers,mileage,price

def estimatePrice(thetaCoef, thetaFix, mileage):
	return thetaCoef * mileage + thetaFix

def costFunction(thetaCoef, thetaFix, mileage, price, m_length):
	cost = 0
	for i in range(m_length):
		f = estimatePrice(thetaCoef, thetaFix, mileage[i])
		cost = cost + (f - price[i]) ** 2
	return cost / (m_length * 2)

def derivatedCoef(thetaCoef, thetaFix, mileage, price, m_length):
	sum = 0
	for mileage_i, price_i in zip(mileage, price):
		est = estimatePrice(thetaCoef, thetaFix, mileage_i)
		print('est1', est)
		sum = sum + (estimatePrice(thetaCoef, thetaFix, mileage_i) - price_i) * mileage_i
	print('sum1', sum)
	print('m_length1', m_length)
	return sum / m_length

def derivatedFix(thetaCoef, thetaFix, mileage, price, m_length):
	sum = 0
	for mileage_i, price_i in zip(mileage, price):
		est = estimatePrice(thetaCoef, thetaFix, mileage_i)
		print('est2', est)
		sum = sum + estimatePrice(thetaCoef, thetaFix, mileage_i) - price_i
	print('sum2', sum)
	print('m_length2', m_length)
	return sum / m_length
			
def linearRegression(mileage, price):
	m_length = mileage.shape[0]
	thetaCoef = 0
	thetaFix = 0
	for _ in range(EPOCHS):
		cost = costFunction(thetaCoef, thetaFix, mileage, price, m_length)
		d_coef = derivatedCoef(thetaCoef, thetaFix, mileage, price, m_length)
		d_fix = derivatedFix(thetaCoef, thetaFix, mileage, price, m_length)
		print('d_coef, d_fix', d_coef, d_fix)
		thetaCoef = thetaCoef - LEARNING_RATE * d_coef
		thetaFix = thetaFix - LEARNING_RATE * d_fix
		print('cost, coef, fix', cost, thetaCoef, thetaFix, '\n')
	return thetaCoef, thetaFix

def train(filePath):
	headers,mileage,price = readCsv(filePath)
	thetaCoef,thetaFix = linearRegression(mileage,price)
	displayPlot(headers,mileage,price, estimatePrice(thetaCoef, thetaFix, mileage))
	
if __name__ == '__main__':
	if (len(sys.argv) <= 1):
		sys.exit('Usage: python train.py <csv file>')
	# train(sys.argv[1])
	train('data.csv')
