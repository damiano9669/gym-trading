import gym
import numpy as np
from gym.spaces import Discrete

from gym_trading.envs.config import currencies
from gym_trading.envs.trading_game import TradingGame


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 cripto_currency='BTC', mode_random=False,
                 n_samples=5000,
                 buy_fee=0.25, sell_fee=0.25,
                 order=100):

        self.mode_random = mode_random
        self.action_space = Discrete(2)

        self.tr_games = []
        self.index = 0

        # if mode random is active we initialize every currency prices
        if self.mode_random:
            for i, cc in enumerate(currencies.keys()):
                tr_game = TradingGame(cc, n_samples, buy_fee, sell_fee, order)
                self.tr_games.append(tr_game)
        else:
            # otherwise only the cripto selected
            tr_game = TradingGame(cripto_currency, n_samples, buy_fee, sell_fee, order)
            self.tr_games.append(tr_game)

        # index of the currency selected for the game
        self.index = 0 if not mode_random else np.random.randint(0, len(currencies.keys()))

        self.get_summary()
        self.get_optimal_plot()

    def get_summary(self):
        for tr in self.tr_games:
            tr.get_summary()

    def get_optimal_plot(self):
        for tr in self.tr_games:
            tr.plot_optimal()

    def get_tr(self):
        return self.tr_games[self.index]

    def step(self, action):

        tr = self.get_tr()

        reward = 0

        # updating the game
        done = tr.step()

        if action == 0:  # BUY
            tr.buy()
        elif action == 1:  # SELL
            if tr.sell():
                reward = tr.get_profit()

        # return observation, reward, done, infos
        return tr.get_current_price(), reward, done, {}

    def reset(self):
        self.index = 0 if not self.mode_random else np.random.randint(0, len(currencies.keys()))
        self.get_tr().reset()
        return self.get_tr().get_current_price()

    def render(self, mode='human', close=False):
        self.get_tr().plot()

    def get_profit(self):
        return self.get_tr().get_profit()
