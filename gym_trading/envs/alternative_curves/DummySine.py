import datetime

import numpy as np

from gym_trading.envs.Utils import range_normalization


def get_dummy_sine(n_samples):
    x = np.arange(0, n_samples / 4, 0.25)

    initial_date = datetime.datetime.today()
    dates = [initial_date - datetime.timedelta(days=x) for x in range(x.shape[0])]
    dates.reverse()

    y = np.sin(2 * x) + np.sin(x) + np.random.normal(0, 0.2, size=x.shape)
    y = range_normalization(y) * 100 + 1

    return {'dates': dates,
            'BTC_prices': y,
            'XRP_prices': y,
            'ETH_prices': y}
