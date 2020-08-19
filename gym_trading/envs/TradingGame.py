import my_utils.math.stats as st
import numpy as np

from gym_trading.envs.Config import url
from gym_trading.envs.DataLoader import DataLoader
from gym_trading.envs.RenderStyle import *
from gym_trading.envs.Trader import Trader


class TradingGame(Trader):

    def __init__(self, n_samples=None, sampling_every=None, stack_size=1, fee=0.25, aav_decay=0.9):
        """

        :param n_samples: number of samples to load from the database
        :param stack_size: size of the observations
        :param fee:
        """
        super().__init__(init_amount=1000.0,
                         init_currency='USD',
                         current_amount=1000.0,
                         current_currency='USD',
                         crypto_currency='BTC',
                         buy_fee=fee,
                         sell_fee=fee)

        self.aav_decay = aav_decay

        self.data = DataLoader(url).data

        if sampling_every is not None:
            new_dates = []
            new_prices = []
            for i in range(len(self.data['dates'])):
                if i % sampling_every == 0:
                    new_dates.append(self.data['dates'][i])
                    new_prices.append(self.data['prices'][i])
            self.data['dates'] = new_dates
            self.data['prices'] = np.asarray(new_prices)

        if n_samples is not None:
            self.data['dates'] = self.data['dates'][-n_samples:]
            self.data['prices'] = self.data['prices'][-n_samples:]

        self.stack_size = stack_size
        self.stack = []
        self.current_day_index = 0

        self.buy_actions = {'dates': [], 'prices': []}
        self.sell_actions = {'dates': [], 'prices': []}

        self.incremental_AAV = 0
        self.N = 0  # number of buy-sell pairs

    def step(self):
        """
            To perform a step:
                - updating current day;
                - checking if this is the last day;
        :return:
        """
        done = False
        self.current_day_index += 1
        if self.current_day_index >= len(self.data['dates']):
            self.current_day_index -= 1
            done = True
        data = self.get_data_now()

        # adding element to the lsit
        self.stack.append(data)
        # checking if the list exceeds the maximum number
        if len(self.stack) > self.stack_size:
            self.stack.pop(0)

        # if we have done we return the stack, also in case of non-full stack
        if done:
            return self.stack, done
        else:
            # otherwise we call a recursion until stack is full
            if len(self.stack) < self.stack_size:
                return self.step()
            else:
                # in case of normal situations, we return the updated stack
                return self.stack, done

    def buy(self):
        data_now = self.get_data_now()
        performed = super(TradingGame, self).buy(data_now['price'])
        if performed:
            self.buy_actions['dates'].append(data_now['date'])
            self.buy_actions['prices'].append(data_now['price'])

    def sell(self):
        data_now = self.get_data_now()
        performed = super(TradingGame, self).sell(data_now['price'])
        if performed:
            self.sell_actions['dates'].append(data_now['date'])
            self.sell_actions['prices'].append(data_now['price'])

            # for the incremental AAV
            self.N += 1  # updating the number of pairs
            x_k = st.add_percentage(self.sell_actions['prices'][-1],
                                    -self.sell_fee) - st.add_percentage(self.buy_actions['prices'][-1],
                                                                        self.buy_fee)
            # incremental mean formula
            self.incremental_AAV = self.aav_decay * self.incremental_AAV + (x_k - self.incremental_AAV) / self.N
        return performed

    def get_profit(self):
        data_now = self.get_data_now()
        return super(TradingGame, self).get_profit(data_now['price'])

    def get_AAV(self):
        return self.incremental_AAV

    def get_data_now(self):
        return {'date': self.data['dates'][self.current_day_index],
                'price': self.data['prices'][self.current_day_index]}

    def plot_chart(self):
        plt.title(f'Total profit: {round(self.get_profit(), 3)} % (fee: {self.buy_fee} %)')

        plt.plot(self.data['dates'], self.data['prices'], alpha=0.7, label='Price', zorder=1)

        plt.scatter(self.buy_actions['dates'],
                    self.buy_actions['prices'],
                    marker='^', c='g', label='BUY', zorder=2)

        plt.scatter(self.sell_actions['dates'],
                    self.sell_actions['prices'],
                    marker='v', c='r', label='SELL', zorder=2)

        plt.xticks(rotation=90)

        plt.ylabel(f'USD/{self.crypto_currency}')
        plt.xlabel('Time')
        plt.legend()

        interval = self.data['dates'][-1] - self.data['dates'][-2]
        initial_date = self.data['dates'][0]
        final_date = self.data['dates'][-1]

        plt.figtext(0.01,
                    0.01,
                    f'Sampling interval: {round(interval.total_seconds() / 60)} minutes\n'
                    f'Total number of days: {round((((final_date - initial_date).total_seconds() / 60) / 60) / 24)}\n'
                    f'Initial date: {initial_date} - Final date: {final_date}',
                    fontsize=10,
                    verticalalignment='bottom')

        plt.show()

    def reset(self):
        self.stack = []
        self.current_day_index = 0

        self.buy_actions = {'dates': [], 'prices': []}
        self.sell_actions = {'dates': [], 'prices': []}

        self.incremental_AAV = 0
        self.N = 0  # number of buy-sell pairs


if __name__ == '__main__':

    game = TradingGame()

    done = False
    while not done:
        data, done = game.step()
        print(data, done)

    game.plot_chart()
