import io
from datetime import datetime

import my_utils.dir_file.files as fl
import numpy as np
import pandas as pd
import requests

from gym_trading.envs.Config import file_csv


class DataLoader:

    def __init__(self, url):
        self.url = url
        self.data = self.get_dates_and_prices()

    def get_dates_and_prices(self):

        if fl.check_if_file_exists(file_csv, create=False):
            data = self.load_data_from_file()
        else:
            data = self.download_data()
            df = pd.DataFrame(data)
            df.to_csv(file_csv)

        dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in data['dates']]
        return {'dates': dates, 'prices': data['prices'].to_numpy(np.float32)}

    def download_data(self):
        print('Downloading dataset...')
        response = requests.get(self.url).content
        data = pd.read_csv(io.StringIO(response.decode('utf-8')))
        return data

    def load_data_from_file(self):
        data = pd.read_csv(file_csv)
        return data
