import pandas as pd
import numpy as np
import plotly.express as px
import ml_tools as tools
from tqdm import tqdm
import argparse
import sys

LEARNING_RATE = 1
EPOCHS = 5000


def display_plot(headers, df, estimate):
    fig = px.scatter(df, x=headers[0], y=headers[1], title="Price / Mileage")
    fig.add_scatter(x=df[headers[0]], y=estimate, mode="lines", name="Regression line")
    fig.show()


def read_csv(file_path):
    headers = pd.read_csv(file_path, nrows=0).columns
    df = pd.read_csv(file_path, usecols=headers)
    return headers, df


def estimate_price(theta_slope, theta_intercept, mileage):
    return theta_slope * mileage + theta_intercept


def cost_function(theta_slope, theta_intercept, mileage, price, m_length):
    cost = 0
    for i in range(m_length):
        f = estimate_price(theta_slope, theta_intercept, mileage[i])
        cost = cost + (f - price[i]) ** 2
    total_cost = 1 / (2 * m_length) * cost
    return total_cost


def compute_gradient(mileage, price, theta_slope, theta_intercept):
    m = len(mileage)
    dj_dslope = 0
    dj_dintercept = 0
    for i in range(m):
        f_slope_intercept = theta_slope * mileage[i] + theta_intercept
        dj_dslope_i = (f_slope_intercept - price[i]) * mileage[i]
        dj_dintercept_i = f_slope_intercept - price[i]
        dj_dslope = dj_dslope + dj_dslope_i
        dj_dintercept = dj_dintercept + dj_dintercept_i
    dj_dslope = dj_dslope / m
    dj_dintercept = dj_dintercept / m
    return dj_dslope, dj_dintercept


def linear_regression(mileage, price, best_learning_rate):
    theta_slope = 0
    theta_intercept = 0
    cost_array = []
    p_history = []
    for _ in range(EPOCHS):
        dj_dslope, dj_dintercept = compute_gradient(
            mileage, price, theta_slope, theta_intercept
        )
        theta_slope = theta_slope - LEARNING_RATE * dj_dslope
        theta_intercept = theta_intercept - LEARNING_RATE * dj_dintercept
        cost_array.append(
            cost_function(theta_slope, theta_intercept, mileage, price, len(mileage))
        )
        p_history.append([theta_slope, theta_intercept])
        if len(cost_array) > 1 and (
            cost_array[-2] < cost_array[-1] or cost_array[-2] - cost_array[-1] < 10e-9
        ):
            if best_learning_rate == False:
                print("Training stopped after {} epochs".format(len(cost_array)))
            break
    return theta_slope, theta_intercept, cost_array


def plot_cost(cost_array):
    fig = px.line(x=np.arange(len(cost_array)), y=cost_array, title="Cost function")
    fig.show()


def save_theta(theta_slope, theta_intercept, np_km, np_price):
    km_min = tools.min_(np_km)
    km_max = tools.max_(np_km)
    price_min = tools.min_(np_price)
    price_max = tools.max_(np_price)
    theta_slope = (theta_slope * (price_max - price_min)) / (km_max - km_min)
    theta_intercept = theta_intercept * (price_max - price_min) + price_min
    np.save("theta.npy", [theta_slope, theta_intercept])


def find_best_learning_rate(theta_slope, theta_intercept):
    global LEARNING_RATE
    best_learning = LEARNING_RATE
    _, _, cost = linear_regression(theta_slope, theta_intercept, True)
    best_cost = cost[-1]
    for i in tqdm(np.arange(1, 2, 0.001)):
        LEARNING_RATE = i
        _, _, cost = linear_regression(theta_slope, theta_intercept, True)
        if cost[-1] < best_cost:
            best_cost = cost[-1]
            best_learning = i
    print("Best learning rate:", best_learning)
    LEARNING_RATE = best_learning


def train_model(file_path, display, cost, learning_rate: bool, accuracy: bool):
    headers, df = read_csv(file_path)
    np_km = df["km"].to_numpy()
    np_price = df["price"].to_numpy()
    norm_x = tools.normalize_array(np_km)
    norm_y = tools.normalize_array(np_price)
    if learning_rate == True:
        find_best_learning_rate(norm_x, norm_y)
    normalized_theta_slope, normalized_theta_intercept, cost_array = linear_regression(
        norm_x, norm_y, False
    )
    if cost == True:
        plot_cost(cost_array)
    if display == True:
        norm_df = pd.DataFrame({"km": norm_x, "price": norm_y})
        display_plot(
            headers,
            norm_df,
            estimate_price(normalized_theta_slope, normalized_theta_intercept, norm_x),
        )
    save_theta(normalized_theta_slope, normalized_theta_intercept, np_km, np_price)
    if accuracy == True:
        print(f"Accuracy: {((1 - cost_array[-1]) * 100):.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a linear regression model")
    parser.add_argument(
        "file_path", metavar="file_path", type=str, help="csv file path"
    )
    parser.add_argument("-d", "--display", help="display plot", action="store_true")
    parser.add_argument(
        "-a",
        "--accuracy",
        help="calculates accuracy",
        action="store_true",
    )
    parser.add_argument("-c", "--cost", help="plot cost function", action="store_true")
    parser.add_argument("-e", "--epochs", help="number of epochs")
    parser.add_argument(
        "-l", "--learning_rate", action="store_true", help="find best learning rate"
    )
    args = parser.parse_args()
    try:
        tools.is_valid_path(args.file_path)
        if args.epochs:
            EPOCHS = int(args.epochs)
        if args.learning_rate:
            LEARNING_RATE = float(args.learning_rate)
    except Exception as e:
        sys.exit(e)
    train_model(
        args.file_path, args.display, args.cost, args.learning_rate, args.accuracy
    )
