import sys
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.1
MAX_ITERATIONS = 100

class point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

def displayPlot(headers,x, y):
	plt.figure(figsize=(10,6))
	plt.xlabel(headers[0])
	plt.ylabel(headers[1])
	plt.plot(x, y, 'ob', label = 'data')
	plt.show()

def readCsv(filePath):
	try:
		f = open(filePath, 'r')
	except OSError:
		print('cannot open', filePath)
		sys.exit(0)
	headers = f.readline().split(',')
	f.close()
	x,y = np.loadtxt(filePath, delimiter=',', unpack=True, skiprows=1, dtype=int)
	return headers,x,y

def linearRegression(x, y):
	theta0 = 0
	theta1 = 0
	print(x, y)
	print(theta0, theta1)

def train(filePath):
	headers,x,y = readCsv(filePath)
	# displayPlot(headers,x,y)
	linearRegression(x,y)
	
if __name__ == '__main__':
	if (len(sys.argv) <= 1):
		sys.exit('Usage: python train.py <csv file>')
	train(sys.argv[1])