import signal
import json

def signal_handler(sig, frame):
    exit("\nVous avez quitté le programme.")

signal.signal(signal.SIGQUIT, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def getTheta():
	try:
		f = open('theta.json', 'r')
	except OSError:
		print('cannot open theta.json')
		exit(0)
	data = json.load(f)
	f.close()
	print(data)
	return data['theta0'], data['theta1']

def predict(input):
	x,y = getTheta()
	print(x + y * input)

if __name__ == '__main__':
	try:
		input = input('Enter a mileage: ')
		predict(int(input))
	except (KeyboardInterrupt, EOFError, SystemExit):
		exit("\nVous avez quitté le programme.")
	except ValueError:
		exit("\nVeuillez entrer un nombre entier.")