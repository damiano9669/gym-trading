import my_utils.math.stats as st

from gym_trading.envs.data_handler import load_data


class TradingGame():

    def __init__(self, n_samples, buy_fee=0.0, sell_fee=0.0):

        self.buy_fee = buy_fee
        self.sell_fee = sell_fee

        # loading data
        self.dates, self.prices = load_data(n_samples)
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
            self.currency = 'BTC'

    def sell(self):
        if self.currency == 'BTC':
            # saving action
            self.sell_actions['x'].append(self.dates[self.status])
            self.sell_actions['y'].append(self.prices[self.status])
            # updating amount
            self.amount = st.add_percentage(self.amount * self.prices[self.status], -self.sell_fee)
            self.currency = 'USD'

    def get_reward(self):
        if self.currency == 'USD':
            # to compute the difference of amount in percentage
            return self.amount / self.init_amount - 1
        else:
            # in this case, before to compute the percentage, we have to do a fake conversion in USD
            return (self.amount * self.prices[self.status]) / self.init_amount - 1

    def get_current_price(self):
        return self.prices[self.status]

    def step(self):
        self.status += 1
        # updating time
        # the game is finished only if we have see all data
        if self.status == self.prices.shape[0] - 1:
            return True
        else:
            return False

    def reset(self):
        self.actions = {'x': [], 'y': [], 'action': []}
        self.amount = self.init_amount = 100  # USD
        self.currency = 'USD'
        self.status = 0  # 0 <= status < len(dates)


if __name__ == '__main__':
    tr_game = TradingGame(100)

    tr_game.reset()

    tr_game.buy()

    tr_game.sell()

    tr_game.step()

    print(tr_game.get_reward())
