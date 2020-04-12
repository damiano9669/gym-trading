import numpy as np
import pandas as pd


def load_data(n_samples=1000):
    data = pd.read_csv('gym_trading/envs/historical_data/BTC_USD.csv')[-n_samples:]
    # data = pd.read_csv('historical_data/BTC_USD.csv')[-n_samples:]

    dates = data['Date']
    high = data['High']
    low = data['Low']

    prices = (high + low) / 2

    return dates.tolist(), prices.to_numpy(dtype=np.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dates, prices = load_data(1000)

    print(prices.shape)

    plt.plot(dates, prices)
    plt.show()
