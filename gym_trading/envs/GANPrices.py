import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


class GANPrices:
    sampling_intervals = {96: {'model_path': 'models/BTC_generator_model_1day',
                               'seed_size': 100,
                               'output_len': 200}}

    def __init__(self, sampling_every):
        self.interval = sampling_every
        if self.interval not in list(self.sampling_intervals.keys()):
            raise Exception(
                f'No model has been trained for the selected sampling interval. '
                f'Please, choose the value of the parameter \'sampling_every\' between: '
                f'{list(self.sampling_intervals.keys())}')

        path = Path(__file__).parent / self.sampling_intervals[self.interval]['model_path']
        self.model = tf.keras.models.load_model(path)

        initial_date = datetime.datetime.today()
        self.dates = [initial_date - datetime.timedelta(days=x) for x in
                      range(self.sampling_intervals[self.interval]['output_len'])]
        self.dates.reverse()

    def get_sample(self):
        seed = tf.random.normal([1, self.sampling_intervals[self.interval]['seed_size']])
        prediction = self.model(seed, training=False)[0]

        # I want only negative trends to increase the complexity for the learner
        while prediction[-1] > prediction[0]:
            seed = tf.random.normal([1, self.sampling_intervals[self.interval]['seed_size']])
            prediction = self.model(seed, training=False)[0]

        # conversion to numpy
        prediction = prediction.numpy()

        return {'dates': self.dates,
                'BTC_prices': prediction,
                'XRP_prices': np.zeros(shape=prediction.shape),
                'ETH_prices': np.zeros(shape=prediction.shape)}
