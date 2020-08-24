import gym
import numpy as np
from gym.spaces import Discrete

from gym_trading.envs.TradingGame import TradingGame


class TradingEnv(gym.Env):

    def __init__(self, n_samples=None, sampling_every=None, random_initial_date=False, stack_size=1, fee=0.25,
                 reward_function='AAV'):
        """
        :param n_samples: Number of total samples.
        :param stack_size: Number of prices to get for every observation.
        :param fee: percentage of the fee for every conversion.
        """
        self.observation_space = np.zeros(shape=(stack_size, 1))
        self.action_space = Discrete(2)  # BUY, SELL
        self.trader = TradingGame(n_samples=n_samples,
                                  sampling_every=sampling_every,
                                  random_initial_date=random_initial_date,
                                  stack_size=stack_size,
                                  fee=fee,
                                  reward_function=reward_function)
        self.reset()

    def step(self, action):
        """

        :param action: 0 (BUY) or 1 (SELL)
        :return: observation, reward, done, infos -> observation can be an unique price or a numpy array of prices.
        """

        observation, done = self.trader.step(action)
        observation = np.array([ob['price'] for ob in observation])

        return observation, self.trader.get_reward(), done, {}

    def reset(self):
        self.trader.reset()
        observation, done = self.trader.step()
        observation = np.array([ob['price'] for ob in observation])
        return observation

    def render(self):
        self.trader.plot_chart()

    def get_profit(self):
        return self.trader.get_profit()
