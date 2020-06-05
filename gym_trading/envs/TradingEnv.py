import gym
from gym.spaces import Discrete

from gym_trading.envs.TradingGame import TradingGame


class TradingEnv(gym.Env):

    def __init__(self, n_samples=None, fee=0.25):
        self.n_samples = n_samples
        self.fee = fee
        self.action_space = Discrete(2)  # BUY, SELL
        self.trader = None
        self.reset()  # to initialize the trader

    def step(self, action):
        reward = 0

        if action == 0:
            self.trader.buy()
        elif action == 1:
            self.trader.sell()
            reward = self.trader.get_profit()

        observation, done = self.trader.step()
        return observation['price'], reward, done, {}

    def reset(self):
        self.trader = TradingGame(n_samples=self.n_samples, fee=self.fee)
        return self.trader.get_data_now()['price']

    def render(self):
        self.trader.plot_chart()

    def get_profit(self):
        return self.trader.get_profit()
