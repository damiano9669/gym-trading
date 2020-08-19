import gym
import numpy as np
from gym.spaces import Discrete

from gym_trading.envs.TradingGame import TradingGame


class TradingEnv(gym.Env):

    def __init__(self, n_samples=None, sampling_every=None, stack_size=1, fee=0.25, aav_decay=0.9):
        """
        :param n_samples: Number of total samples.
        :param stack_size: Number of prices to get for every observation.
        :param fee: percentage of the fee for every conversion.
        """
        self.n_samples = n_samples
        self.sampling_every = sampling_every
        self.stack_size = stack_size
        self.fee = fee
        self.observation_space = np.zeros(shape=(stack_size,))
        self.action_space = Discrete(2)  # BUY, SELL
        self.trader = TradingGame(n_samples=self.n_samples,
                                  sampling_every=self.sampling_every,
                                  stack_size=self.stack_size,
                                  fee=self.fee,
                                  aav_decay=aav_decay)
        self.reset()

    def step(self, action):
        """

        :param action:
        :return: observation, reward, done, infos -> observation can be an unique price or a numpy array of prices.
        """

        if action == 0:
            self.trader.buy()
        elif action == 1:
            self.trader.sell()

        observation, done = self.trader.step()
        observation = np.array([ob['price'] for ob in observation])

        return observation, self.trader.get_AAV(), done, {}

    def reset(self):
        self.trader.reset()
        observation, done = self.trader.step()
        observation = np.array([ob['price'] for ob in observation])
        return observation

    def render(self):
        self.trader.plot_chart()

    def get_profit(self):
        return self.trader.get_profit()
