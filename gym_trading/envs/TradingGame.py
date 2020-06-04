import matplotlib.pyplot as plt

from gym_trading.envs.Config import url
from gym_trading.envs.DataLoader import DataLoader
from gym_trading.envs.Trader import Trader


class TradingGame(Trader):

    def __init__(self, fee=0.25):
        super().__init__(init_amount=1000.0,
                         init_currency='USD',
                         current_amount=1000.0,
                         current_currency='USD',
                         crypto_currency='BTC',
                         buy_fee=fee,
                         sell_fee=fee)

        self.data = DataLoader(url).data
        self.current_day_index = 0

        self.buy_actions = {'dates': [], 'prices': []}
        self.sell_actions = {'dates': [], 'prices': []}

    def step(self):
        done = False
        self.current_day_index += 1
        if self.current_day_index >= len(self.data['dates']):
            self.current_day_index -= 1
            done = True
        data = self.get_data_now()
        return data, done

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
