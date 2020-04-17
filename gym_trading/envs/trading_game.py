import my_utils.math.stats as st
import numpy as np

from gym_trading.envs.data_handler import get_min_max_relatives, get_nearest_minmax
from gym_trading.envs.data_loader import load_data


class TradingGame():

    def __init__(self, cripto_currency='BTC', n_samples=10000, buy_fee=0.25, sell_fee=0.25, order=100):

        self.cripto_currency = cripto_currency
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.order = order

        # loading data
        self.dates, self.prices = load_data(currency=self.cripto_currency, n_samples=n_samples)
        # to compute the reward we'll use the min/max relative points
        self.min_relatives, self.max_relatives = get_min_max_relatives(self.prices, order=self.order)

        # to store the points in which the agent decides to convert
        self.sell_actions = {'x': [], 'y': []}
        self.buy_actions = {'x': [], 'y': []}

        # initialization of the amount. The value is symbolic, only to compute the percentage reward
        self.amount = self.init_amount = 100  # USD
        # current currency
        self.currency = 'USD'

        # current status
        self.status = 0  # 0 <= status < len(dates)

    def buy(self):
        if self.currency == 'USD':
            # saving action
            self.buy_actions['x'].append(self.dates[self.status])
            self.buy_actions['y'].append(self.prices[self.status])
            # updating amount
            self.amount = st.add_percentage(self.amount / self.prices[self.status], -self.buy_fee)
            self.currency = self.cripto_currency
        return self.get_reward()[0]

    def sell(self):
        if self.currency == self.cripto_currency:
            # saving action
            self.sell_actions['x'].append(self.dates[self.status])
            self.sell_actions['y'].append(self.prices[self.status])
            # updating amount
            self.amount = st.add_percentage(self.amount * self.prices[self.status], -self.sell_fee)
            self.currency = 'USD'
        return self.get_reward()[1]

    def get_reward(self):
        min_nearest, max_nearest = get_nearest_minmax(self.status,
                                                      self.min_relatives[0], self.max_relatives[0],
                                                      self.prices)

        # if we are selling we return a positive reward only if the choice is more near to the local minimum
        # otherwise viceversa
        if np.abs(self.status - min_nearest[0]) < np.abs(self.status - max_nearest[0]):
            return 1, -1  # buying reward, selling reward
        else:
            return -1, 1  # buying reward, selling reward

    def get_percentage_profit(self):
        if self.currency == 'USD':
            # to compute the difference of amount in percentage
            return self.amount / self.init_amount - 1
        else:
            # in this case, before to compute the percentage, we have to do a fake conversion in USD
            return (self.amount * self.prices[self.status]) / self.init_amount - 1

    def get_current_price(self):
        return (self.dates[self.status], self.prices[self.status])

    def step(self):
        self.status += 1
        # updating time
        # the game is finished only if we have see all data
        if self.status == self.prices.shape[0] - 1:
            return True
        else:
            return False

    def reset(self):
        self.sell_actions = {'x': [], 'y': []}
        self.buy_actions = {'x': [], 'y': []}
        self.amount = self.init_amount = 100  # USD
        self.currency = 'USD'
        self.status = 0  # 0 <= status < len(dates)

    def max_profit(self):
        done = False
        while not done:
            if self.status in self.min_relatives[0]:
                self.buy()
            elif self.status in self.max_relatives[0]:
                self.sell()

            done = self.step()

        profit = self.get_percentage_profit()
        self.reset()
        return profit

    def get_summary(self):
        summary = f'--------------------------------------------------------\n' \
                  f'Cripto currency: {self.cripto_currency}\n' \
                  f'Initial amount: {self.init_amount} {self.currency}\n' \
                  f'Number of samples: {self.prices.shape[0]}\n' \
                  f'Initial date: {self.dates[0]}\tLast date: {self.dates[-1]}\n' \
                  f'Fees >> buy: -{self.buy_fee} %\tsell: -{self.sell_fee} %\n' \
                  f'min/max relatives order: {self.order}\n' \
                  f'Max profit (with this configurations): {self.max_profit()} %\n' \
                  f'--------------------------------------------------------\n'
        print(summary)


if __name__ == '__main__':
    tr = TradingGame('BTC', 10000, 0.25, 0.25, 20)

    tr.get_summary()
