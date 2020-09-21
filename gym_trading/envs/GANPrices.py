import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


class GANPrices:
    sampling_intervals = {192: {'model_path': 'models/BTC_generator_model_2days',
                                'seed_size': 100,
                                'output_len': 300}}

    def __init__(self, sampling_interval):
        self.interval = sampling_interval
        if self.interval not in list(self.sampling_intervals.keys()):
            raise Exception(
                f'No model has been trained for the selected sampling_interval. '
                f'Please, choose between: {list(self.sampling_intervals.keys())}')

        path = Path(__file__).parent / self.sampling_intervals[self.interval]['model_path']
        self.model = tf.keras.models.load_model(path)

        initial_date = datetime.datetime.today()
        self.dates = [initial_date - datetime.timedelta(days=x) for x in
                      range(self.sampling_intervals[self.interval]['output_len'])]
        self.dates.reverse()

    def get_sample(self):
        seed = tf.random.normal([1, self.sampling_intervals[self.interval]['seed_size']])
        prediction = self.model(seed, training=False)[0]

        # conversion to numpy
        prediction = prediction.numpy()

        return {'dates': self.dates,
                'BTC_prices': prediction,
                'XRP_prices': np.zeros(shape=prediction.shape),
                'ETH_prices': np.zeros(shape=prediction.shape)}
