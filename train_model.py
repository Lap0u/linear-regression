import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ML_kit.array_tools import normalize_array

LEARNING_RATE = 0.035
EPOCHS = 2200

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

def estimatePrice(thetaSlope, thetaIntercept, mileage):
	return thetaSlope * mileage + thetaIntercept

def costFunction(thetaSlope, thetaIntercept, mileage, price, m_length):
	cost = 0
	for i in range(m_length):
		f = estimatePrice(thetaSlope, thetaIntercept, mileage[i])
		cost = cost + (f - price[i]) ** 2
	return cost / (m_length * 2)

def derivatedSlope(thetaSlope, thetaIntercept, mileage, price, m_length):
	sum = 0
	for mileage_i, price_i in zip(mileage, price):
		est = estimatePrice(thetaSlope, thetaIntercept, mileage_i)
		print('est1', est)
		sum = sum + (estimatePrice(thetaSlope, thetaIntercept, mileage_i) - price_i) * mileage_i
	print('sum1', sum)
	print('m_length1', m_length)
	return sum / m_length

def derivatedFix(thetaSlope, thetaIntercept, mileage, price, m_length):
	sum = 0
	for mileage_i, price_i in zip(mileage, price):
		est = estimatePrice(thetaSlope, thetaIntercept, mileage_i)
		print('est2', est)
		sum = sum + estimatePrice(thetaSlope, thetaIntercept, mileage_i) - price_i
	print('sum2', sum)
	print('m_length2', m_length)
	return sum / m_length
			
def linearRegression(mileage, price):
	m_length = mileage.shape[0]
	costArray = []
	thetaSlope = 0
	thetaIntercept = 0
	for _ in range(EPOCHS):
		cost = costFunction(thetaSlope, thetaIntercept, mileage, price, m_length)
		costArray.append(cost)
		d_Slope = derivatedSlope(thetaSlope, thetaIntercept, mileage, price, m_length)
		d_fix = derivatedFix(thetaSlope, thetaIntercept, mileage, price, m_length)
		thetaSlope = thetaSlope - LEARNING_RATE * d_Slope
		thetaIntercept = thetaIntercept - LEARNING_RATE * d_fix
		if (len(costArray) > 1 and (costArray[-2] < costArray[-1] or costArray[-2] - costArray[-1] < 10e-9)):
			break
	print('thetaSlope', thetaSlope)
	print('thetaIntercept', thetaIntercept)
	print('cost', cost)
	print('epochs', len(costArray))
	return thetaSlope, thetaIntercept

def train_model(filePath):
	headers,mileage,price = readCsv(filePath)
	normalizedMileage = normalize_array(mileage)
	normalizedPrice = normalize_array(price)
	thetaSlope,thetaIntercept = linearRegression(normalizedMileage,normalizedPrice)
	displayPlot(headers,normalizedMileage,normalizedPrice, estimatePrice(thetaSlope, thetaIntercept, normalizedMileage))
	
if __name__ == '__main__':
	if (len(sys.argv) <= 1):
		sys.exit('Usage: python train.py <csv file>')
	train_model(sys.argv[1])
