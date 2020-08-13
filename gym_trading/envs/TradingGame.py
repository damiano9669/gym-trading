import matplotlib.pyplot as plt
import my_utils.math.stats as st

from gym_trading.envs.Config import url
from gym_trading.envs.DataLoader import DataLoader
from gym_trading.envs.Trader import Trader


class TradingGame(Trader):

    def __init__(self, n_samples=None, stack_size=1, fee=0.25):
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

        self.data = DataLoader(url).data
        if n_samples is not None:
            self.data['dates'] = self.data['dates'][-n_samples:]
            self.data['prices'] = self.data['prices'][-n_samples:]

        self.stack_size = stack_size
        self.stack = []
        self.current_day_index = 0

        self.buy_actions = {'dates': [], 'prices': []}
        self.sell_actions = {'dates': [], 'prices': []}

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
        self.buy_actions['dates'].append(data_now['date'])
        self.buy_actions['prices'].append(data_now['price'])
        return super(TradingGame, self).buy(data_now['price'])

    def sell(self):
        data_now = self.get_data_now()
        self.sell_actions['dates'].append(data_now['date'])
        self.sell_actions['prices'].append(data_now['price'])
        return super(TradingGame, self).sell(data_now['price'])

    def get_profit(self):
        data_now = self.get_data_now()
        return super(TradingGame, self).get_profit(data_now['price'])

    def get_AAV(self):
        N = len(self.buy_actions['dates']) + len(self.sell_actions['dates'])
        aav = 0
        for price_b, price_s in zip(self.buy_actions['prices'],
                                    self.sell_actions['prices'] if N % 2 == 0 else self.sell_actions['prices'][:-1]):
            aav += st.add_percentage(price_s, -self.sell_fee) - st.add_percentage(price_b, self.buy_fee)
        N = N if N % 2 == 0 else N - 1
        if N == 0:
            return 0
        return aav / N

    def get_data_now(self):
        return {'date': self.data['dates'][self.current_day_index],
                'price': self.data['prices'][self.current_day_index]}

    def plot_chart(self):
        plt.plot(self.data['dates'], self.data['prices'], label='Price')

        plt.scatter(self.buy_actions['dates'],
                    self.buy_actions['prices'],
                    marker='^', c='g', label='BUY')

        plt.scatter(self.sell_actions['dates'],
                    self.sell_actions['prices'],
                    marker='v', c='r', label='SELL')

        plt.xticks(rotation=90)

        plt.ylabel(f'USD/{self.crypto_currency}')
        plt.xlabel('Date')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    game = TradingGame()

    done = False
    while not done:
        data, done = game.step()
        print(data, done)

    game.plot_chart()
