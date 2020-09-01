import gym
import numpy as np
from gym.spaces import Discrete

from gym_trading.envs.TradingGame import TradingGame


class TradingEnv(gym.Env):

    def __init__(self, n_samples=None, sampling_every=None, random_initial_date=False, stack_size=1, fee=0.25,
                 reward_function='AAV', endurance_mode=False, normalize_observation=False):
        """
        :param n_samples: Number of total samples.
        :param stack_size: Number of prices to get for every observation.
        :param fee: percentage of the fee for every conversion.
        """
        self.endurance_mode = endurance_mode
        self.normalize = normalize_observation
        self.observation_space = np.zeros(shape=(stack_size, 3))
        self.action_space = Discrete(6)
        # 0 (BUY_BTC),  1 (SELL_BTC),
        # 2 (BUY_XRP), 3 (SELL_XRP),
        # 4 (BUY_ETH), 5 (SELL_ETH)
        self.trader = TradingGame(n_samples=n_samples,
                                  sampling_every=sampling_every,
                                  random_initial_date=random_initial_date,
                                  stack_size=stack_size,
                                  fee=fee,
                                  reward_function=reward_function)
        self.reset()

    def step(self, action):
        """

        :param action: 0 (BUY_BTC),  1 (SELL_BTC), 2 (BUY_XRP), 3 (SELL_XRP), 4 (BUY_ETH), 5 (SELL_ETH)
        :return: observation, reward, done, infos -> observation can be an unique price or a numpy array of prices.
        """

        observation, done = self.trader.step(action)
        observation = self.clean_observation(observation)

        reward = self.trader.get_reward()
        if len(self.trader.rewards['rewards']) > 2:
            if self.endurance_mode and self.trader.rewards['rewards'][-1] < self.trader.rewards['rewards'][-2]:
                done = True  # stopping condition in case of loosing money

        if self.normalize:
            observation = self.normalize_observation(observation)

        return observation, reward, done, {}

    def reset(self):
        self.trader.reset()
        observation, done = self.trader.step()
        observation = self.clean_observation(observation)
        return observation

    def render(self):
        self.trader.plot_chart()

    def get_profit(self):
        return self.trader.get_profit()

    def clean_observation(self, observation):
        crypto_obs = []
        for key in list(self.trader.data.keys())[1:]:
            crypto_stack = []
            for ob in observation:
                crypto_stack.append(ob[key.replace('s', '')])
            crypto_obs.append(np.asarray(crypto_stack))

        return np.stack(crypto_obs, axis=-1)

    def normalize_observation(self, x):
        x_mean = np.mean(x)
        x_std = np.std(x)
        return (x - x_mean) / x_std
