import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Discrete

from gym_trading.envs.config import currencies
from gym_trading.envs.trading_game import TradingGame


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cripto_currency='BTC', mode_random=False, n_samples=5000, buy_fee=0.25, sell_fee=0.25,
                 order=5):

        self.mode_random = mode_random
        self.action_space = Discrete(3)
        self.tr_games = []

        self.index = 0

        # if mode random is active we initialize every currency prices
        if self.mode_random:
            for i, cc in enumerate(currencies.keys()):
                tr_game = TradingGame(cc, n_samples, buy_fee, sell_fee, order)
                tr_game.get_summary()
                self.tr_games.append(tr_game)
                self.index = i
                self.render_optimal()
        else:
            # otherwise only the cripto selected
            tr_game = TradingGame(cripto_currency, n_samples, buy_fee, sell_fee, order)
            tr_game.get_summary()
            self.tr_games.append(tr_game)
            self.render_optimal()

        # index of the currency selected for the game
        self.index = 0 if not mode_random else np.random.randint(0, len(currencies.keys()))

    def step(self, action):

        # updating the game
        done = self.tr_games[self.index].step()

        if action == 0:
            # in this case we keep the current currency
            reward = 0
        elif action == 1:
            # in this case we BUY USD
            reward = self.tr_games[self.index].buy()
        elif action == 2:
            # in this case we SELL BTC
            reward = self.tr_games[self.index].sell()

        # return observation, reward, done, infos
        return self.tr_games[self.index].get_current_price(), reward, done, None

    def reset(self):
        self.index = 0 if not self.mode_random else np.random.randint(0, len(currencies.keys()))
        self.tr_games[self.index].reset()
        return self.tr_games[self.index].get_current_price()

    def render(self, mode='human', close=False):
        idx = self.tr_games[self.index].status

        plt.clf()

        plt.plot(self.tr_games[self.index].dates[:idx], self.tr_games[self.index].prices[:idx], label='Price')

        plt.scatter(self.tr_games[self.index].buy_actions['x'][:idx],
                    self.tr_games[self.index].buy_actions['y'][:idx],
                    marker='^', c='g', label='BUY')

        plt.scatter(self.tr_games[self.index].sell_actions['x'][:idx],
                    self.tr_games[self.index].sell_actions['y'][:idx],
                    marker='v', c='r', label='SELL')

        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        plt.ylabel(f'USD/{self.get_current_cripto()}')
        plt.xlabel('Date')
        plt.legend()
        plt.show()

    def render_optimal(self):
        plt.clf()

        plt.plot(self.tr_games[self.index].dates, self.tr_games[self.index].prices, label='Price')

        plt.scatter(self.tr_games[self.index].min_relatives[0],
                    self.tr_games[self.index].min_relatives[1],
                    marker='^', c='g', label='BUY')

        plt.scatter(self.tr_games[self.index].max_relatives[0],
                    self.tr_games[self.index].max_relatives[1],
                    marker='v', c='r', label='SELL')

        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        plt.ylabel(f'USD/{self.get_current_cripto()}')
        plt.xlabel('Date')
        plt.legend()
        plt.show()

    def get_percentage_profit(self):
        return self.tr_games[self.index].get_percentage_profit()

    def get_current_cripto(self):
        return self.tr_games[self.index].cripto_currency
