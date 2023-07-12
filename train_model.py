import sys
import pandas as pd
import numpy as np
import plotly.express as px
import ml_tools as tools
import argparse

LEARNING_RATE = 0.035
EPOCHS = 5000


def display_plot(headers, df, estimate):
    fig = px.scatter(df, x=headers[0], y=headers[1], title='Price / Mileage')
    fig.add_scatter(x=df[headers[0]], y=estimate,
                    mode='lines', name='Regression line')
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


def linear_regression(mileage, price):
    theta_slope = 0
    theta_intercept = 0
    cost_array = []
    p_history = []
    for i in range(EPOCHS):
        dj_dslope, dj_dintercept = compute_gradient(
            mileage, price, theta_slope, theta_intercept)
        theta_slope = theta_slope - LEARNING_RATE * dj_dslope
        theta_intercept = theta_intercept - LEARNING_RATE * dj_dintercept
        cost_array.append(cost_function(
            theta_slope, theta_intercept, mileage, price, len(mileage)))
        p_history.append([theta_slope, theta_intercept])
        if (len(cost_array) > 1 and (cost_array[-2] < cost_array[-1] or cost_array[-2] - cost_array[-1] < 10e-9)):
            print("Cout stabilisé à l'epoch", i)
            break
    return theta_slope, theta_intercept, cost_array


def plot_cost(cost_array):
    fig = px.line(x=np.arange(len(cost_array)),
                  y=cost_array, title='Cost function')
    fig.show()


def train_model(file_path):
    headers, df = read_csv(file_path)
    np_km = df['km'].to_numpy()
    np_price = df['price'].to_numpy()
    norm_x = tools.normalize_array(np_km)
    norm_y = tools.normalize_array(np_price)
    normalized_theta_slope, normalized_theta_intercept, cost_array = linear_regression(
        norm_x, norm_y)
    denormalized_theta_intercept, denormalized_theta_slope = tools.denormalize_theta(
        normalized_theta_intercept, normalized_theta_slope, np_km, np_price)
    plot_cost(cost_array)
    norm_df = pd.DataFrame({'km': norm_x, 'price': norm_y})
    display_plot(headers, norm_df, estimate_price(
        normalized_theta_slope, normalized_theta_intercept, norm_x))
    display_plot(headers, df, estimate_price(
        denormalized_theta_slope, denormalized_theta_intercept, np_km))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a linear regression model')
    parser.add_argument('file_path', metavar='file_path',
                        type=str, help='csv file path')
    args = parser.parse_args()
    try:
        tools.is_valid_path(args.file_path)
    except Exception as e:
        sys.exit(e)
    train_model(args.file_path)
