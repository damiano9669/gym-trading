import io

import numpy as np
import pandas as pd
import requests

from gym_trading.envs.config import currencies


def load_data(currency='BTC', n_samples=10000):
    response = requests.get(currencies[currency]).content
    data = pd.read_csv(io.StringIO(response.decode('utf-8')))
    data = data.iloc[::-1]
    data = data[-n_samples:]

    dates = data['Date']
    high = data['High']
    low = data['Low']

    prices = (high + low) / 2

    return dates.tolist(), prices.to_numpy(dtype=np.float64)


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def plot(currency):
        dates, prices = load_data(currency=currency)
        print(f'{dates[0]}\t{dates[-1]}\t{len(dates)}')
        prices = prices / max(prices)
        plt.plot(prices, label=currency)


    plot('BTC')
    plot('ETH')
    plot('LTC')
    plot('NEO')

    plt.legend()
    plt.show()
