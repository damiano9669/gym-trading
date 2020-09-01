import io
import os
from datetime import datetime

import my_utils.dir_file.directories as dr
import my_utils.dir_file.files as fl
import numpy as np
import pandas as pd
import requests

from gym_trading.envs import Config


def load_data_from_file(file_csv):
    """
        load data from file csv
    :param file_csv:
    :return:
    """
    data = pd.read_csv(file_csv)
    return data


def download_data(url):
    """
        download data from my server altervista
    :param url:
    :return:
    """
    print(f'Downloading dataset from {url}')
    response = requests.get(url).content
    data = pd.read_csv(io.StringIO(response.decode('utf-8')))
    return data


def get_dates_and_prices(url, file_csv_name):
    """
        checks if data already downloaded, otherwise download
    :param url:
    :param file_csv_name:
    :return:
    """
    dr.check_if_dir_exists('datasets', create=True)

    path_file_csv = os.path.join('datasets', file_csv_name)

    if fl.check_if_file_exists(path_file_csv, create=False):
        data = load_data_from_file(path_file_csv)
    else:
        data = download_data(url)
        df = pd.DataFrame(data)
        df.to_csv(path_file_csv)

    dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in data['dates']]
    return {'dates': dates, 'prices': data['prices'].to_numpy(np.float32)}


def get_all_data():
    """
        this is the final function that uses the previous.
        It returns the dictionary containing all data:
            {'dates': [], 'BTC_prices': [], 'XRP_prices': [], 'ETH_prices': []}
    :return:
    """
    BTC_data = get_dates_and_prices(Config.cryptos['BTC']['url'], Config.cryptos['BTC']['csv'])
    XRP_data = get_dates_and_prices(Config.cryptos['XRP']['url'], Config.cryptos['XRP']['csv'])
    ETH_data = get_dates_and_prices(Config.cryptos['ETH']['url'], Config.cryptos['ETH']['csv'])
    return {'dates': BTC_data['dates'],
            'BTC_prices': BTC_data['prices'],
            'XRP_prices': XRP_data['prices'],
            'ETH_prices': ETH_data['prices']}
