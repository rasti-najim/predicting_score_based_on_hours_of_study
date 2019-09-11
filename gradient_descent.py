import numpy as np
import pandas as pd


def calculate_error_function(b, m, points):
    totall_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totall_error += (y - (m * x + b)) ** 2
    return totall_error/float(len(points))


def gradient_descent_fun(current_b, current_m, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((current_m * x) + current_b))
        m_gradient += -(2/N) * x * (y - ((current_m * x) + current_b))
    new_b = current_b - (learning_rate * b_gradient)
    new_m = current_m - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(initial_b, initial_m, points, learning_rate, num_iterations):
    b = initial_b
    m = initial_m
    for i in range(num_iterations):
        b, m = gradient_descent_fun(b, m, np.array(points), learning_rate)
    return [b, m]


def fit():
    points = np.genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    # print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(
    #     initial_b, initial_m, calculate_error_function(initial_b, initial_m, points)))
    # print("Running...")
    b, m = gradient_descent_runner(
        initial_b, initial_m, points, learning_rate, num_iterations)
    # print("After {0} iterations b = {1}, m = {2}, error = {3}".format(
    #     num_iterations, b, m, calculate_error_function(b, m, points)))
    predict(b, m)


def predict(b, m):
    x = int(input("Hours of study: "))
    y = m * x + b
    print("Your score based on {0} hours of study is {1}".format(x, y))


if __name__ == "__main__":
    fit()
