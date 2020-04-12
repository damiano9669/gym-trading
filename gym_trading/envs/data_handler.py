import io

import numpy as np
import pandas as pd
import requests


def load_data(n_samples=1000):
    url = 'https://blackpixel.altervista.org/btc_history/BTC_USD.csv'
    response = requests.get(url).content
    data = pd.read_csv(io.StringIO(response.decode('utf-8')))[-n_samples:]

    dates = data['Date']
    high = data['High']
    low = data['Low']

    prices = (high + low) / 2

    return dates.tolist(), prices.to_numpy(dtype=np.float64)
