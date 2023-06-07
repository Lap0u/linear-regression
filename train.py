import sys
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.000001
EPOCHS = 5

def displayPlot(headers,mileage, price):
	plt.figure(figsize=(10,6))
	plt.xlabel(headers[0])
	plt.ylabel(headers[1])
	plt.plot(mileage, price, 'ob', label = 'data')
	plt.show()

def readCsv(filePath):
	try:
		f = open(filePath, 'r')
	except OSError:
		print('cannot open', filePath)
		sys.exit(0)
	headers = f.readline().split(',')
	f.close()
	mileage,price= np.loadtxt(filePath, delimiter=',', unpack=True, skiprows=1, dtype=int)
	return headers,mileage,price

def costFunction(thetaCoef, thetaFix, mileage, price, m_length):
	cost = 0
	for i in range(m_length):
		f = thetaCoef * mileage[i] + thetaFix
		cost = cost + (f - price[i]) ** 2
	total_cost = 1 / (2 * m_length) * cost
	return total_cost

def derivatedFix(thetaCoef, thetaFix, mileage, price, m_length):
	sum = 0
	for tempmileage, tempprice in zip(mileage, price):
		sum = sum + estimatePrice(thetaCoef, thetaFix, tempmileage) - tempprice
	return sum / m_length

def derivatedCoef(thetaCoef, thetaFix, mileage, price, m_length):
	sum = 0
	for tempmileage, tempprice in zip(mileage, price):
		sum = sum + (estimatePrice(thetaCoef, thetaFix, tempmileage) - tempprice) * tempmileage
	return sum / m_length

def estimatePrice(thetaCoef, thetaFix, mileage):
	return thetaCoef * mileage + thetaFix
      
def linearRegression(mileage, price):
	m_length = mileage.shape[0]
	thetaCoef = 0
	thetaFix = 0
	for _ in range(EPOCHS):
		cost = costFunction(thetaCoef, thetaFix, mileage, price, m_length)
		d_coef = derivatedCoef(thetaCoef, thetaFix, mileage, price, m_length)
		d_fix = derivatedFix(thetaCoef, thetaFix, mileage, price, m_length)
		thetaCoef = thetaCoef - LEARNING_RATE * d_coef
		thetaFix = thetaFix - LEARNING_RATE * d_fix
		print('cost, coef, fix', cost, thetaCoef, thetaFix)

def train(filePath):
	headers,mileage,price = readCsv(filePath)
	# displayPlot(headers,mileage,price)
	linearRegression(mileage,price)
	
if __name__ == '__main__':
	if (len(sys.argv) <= 1):
		sys.exit('Usage: python train.py <csv file>')
	train(sys.argv[1])