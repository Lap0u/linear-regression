import signal
import numpy as np


def signal_handler(sig, frame):
    exit("\nVous avez quitté le programme.")


signal.signal(signal.SIGQUIT, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def get_theta():
    try:
        thetas = np.load('theta.npy')
    except FileNotFoundError:
        exit("Le modèle n'a pas été entraîné.")
    print("theta_slope:", thetas[0])
    print("theta_intercept:", thetas[1])
    return thetas[0], thetas[1]


def predict(input):
    x, y = get_theta()
    print(x * input + y)


if __name__ == '__main__':
    try:
        input = input('Enter a mileage: ')
        predict(int(input))
    except (KeyboardInterrupt, EOFError, SystemExit):
        exit("\nVous avez quitté le programme.")
    except ValueError:
        exit("\nVeuillez entrer un nombre entier.")
