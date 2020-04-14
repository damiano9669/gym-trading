import gym
import matplotlib.pyplot as plt
from gym.spaces import Discrete

from gym_trading.envs.trading_game import TradingGame


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_samples=730, buy_fee=0.5, sell_fee=0.5):  # 2 years
        self.action_space = Discrete(3)
        self.tr_game = TradingGame(n_samples, buy_fee, sell_fee)

    def step(self, action):

        # updating the game
        done = self.tr_game.step()

        if action == 0:
            # in this case we keep the current currency
            reward = 0
        elif action == 1:
            # in this case we BUY USD
            reward = self.tr_game.buy()
        elif action == 2:
            # in this case we SELL BTC
            reward = self.tr_game.sell()

        # return observation, reward, done, infos
        return self.tr_game.get_current_price(), reward, done, None

    def reset(self):
        self.tr_game.reset()
        return self.tr_game.get_current_price()

    def render(self, mode='human', close=False):
        idx = self.tr_game.status

        plt.clf()

        plt.plot(self.tr_game.dates[:idx], self.tr_game.prices[:idx], label='Price')

        plt.scatter(self.tr_game.buy_actions['x'][:idx],
                    self.tr_game.buy_actions['y'][:idx],
                    marker='^', c='g', label='BUY')

        plt.scatter(self.tr_game.sell_actions['x'][:idx],
                    self.tr_game.sell_actions['y'][:idx],
                    marker='v', c='r', label='SELL')

        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        plt.ylabel('USD/BTC')
        plt.xlabel('Date')
        plt.legend()
        plt.show()

    def get_total_reward(self):
        return self.tr_game.get_percentage_profit()
