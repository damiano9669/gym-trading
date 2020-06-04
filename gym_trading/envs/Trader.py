import my_utils.math.stats as st


class Trader():

    def __init__(self,
                 init_amount=1000.0,
                 init_currency='USD',
                 current_amount=1000.0,
                 current_currency='USD',
                 crypto_currency='BTC',
                 buy_fee=0.25,
                 sell_fee=0.25):

        self.init_amount = init_amount
        self.init_currency = init_currency
        self.current_amount = current_amount
        self.current_currency = current_currency
        self.crypto_currency = crypto_currency

        self.buy_fee = buy_fee
        self.sell_fee = sell_fee

        self.amounts = [self.init_amount]

    def get_profit(self, price):
        if self.current_currency == self.init_currency:
            # to compute the difference of amount in percentage
            return (self.current_amount / self.init_amount - 1) * 100
        else:
            # in this case, before to compute the percentage, we have to do a fake conversion in USD
            return ((self.current_amount * price) / self.init_amount - 1) * 100

    def buy(self, price):
        if self.current_currency == self.init_currency:
            self.current_amount = st.add_percentage(self.current_amount / price, -self.buy_fee)
            self.current_currency = self.crypto_currency
            return True
        return False

    def sell(self, price):
        if self.current_currency == self.crypto_currency:
            self.current_amount = st.add_percentage(self.current_amount * price, -self.sell_fee)
            self.current_currency = self.init_currency
            self.amounts.append(self.current_amount)
            return True
        return False
