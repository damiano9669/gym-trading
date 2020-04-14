import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from gym_trading.envs.data_loader import load_data


def get_min_max_relatives(data, order=5):
    """
    :param data: list of values
    :param order: window
    :return: 2 tuples containing the indices and the values
    """
    min_relatives = np.array(argrelextrema(data, np.less_equal, order=order)).T[..., 0]
    max_relatives = np.array(argrelextrema(data, np.greater_equal, order=order)).T[..., 0]

    return (min_relatives, data[min_relatives]), (max_relatives, data[max_relatives])


def get_nearest_minmax(current_index, min_indices, max_indices, data):
    """

    :param current_index: point to study
    :param min_indices: indices of min values
    :param max_indices: indices of max values
    :param data: original values
    :return: nearest points
    """
    min_nearest = min_indices[(np.abs(min_indices - current_index)).argmin()]
    max_nearest = max_indices[(np.abs(max_indices - current_index)).argmin()]

    return (min_nearest, data[min_nearest]), (max_nearest, data[max_nearest])


if __name__ == '__main__':
    dates, prices = load_data()

    min_relatives, max_relatives = get_min_max_relatives(prices)

    point = 500
    price_point = prices[point]
    min_nearest, max_nearest = get_nearest_minmax(point, min_relatives[0], max_relatives[0], prices)

    print(min_nearest)
    print(max_nearest)

    print('Buying distance:', np.abs(point - min_nearest[0]))
    print('Selling distance:', np.abs(point - max_nearest[0]))

    plt.plot(prices)

    plt.scatter(min_relatives[0], min_relatives[1], marker='^', c='g')
    plt.scatter(max_relatives[0], max_relatives[1], marker='v', c='r')

    plt.scatter(point, price_point, marker='o', c='y')
    plt.scatter(min_nearest[0], min_nearest[1], marker='^', c='y')
    plt.scatter(max_nearest[0], max_nearest[1], marker='v', c='y')
    plt.show()
