import copy

import numpy as np

from gym_trading.envs.Config import url
from gym_trading.envs.DataLoader import DataLoader
from gym_trading.envs.RenderStyle import *
from gym_trading.envs.Trader import Trader
from gym_trading.envs.reward_functions.AAV import get_AAV
from gym_trading.envs.reward_functions.AllenKarjalainen import get_AllenKarjalainen_excess_return
from gym_trading.envs.reward_functions.ROI import get_ROI


class TradingGame(Trader):
    reward_functions = {'ROI': get_ROI,
                        'AAV': get_AAV,
                        'Allen_and_Karjalainen': get_AllenKarjalainen_excess_return}

    def __init__(self, n_samples=None, sampling_every=None, random_initial_date=False, stack_size=1, fee=0.25,
                 reward_function='AAV'):
        """

        :param n_samples: number of samples to load from the database
        :param stack_size: size of the observations
        :param fee:
        """
        super().__init__(init_amount=1000.0,
                         init_currency='USD',
                         crypto_currency='BTC',
                         buy_fee=fee,
                         sell_fee=fee)

        self.n_samples = n_samples
        self.sampling_every = sampling_every
        self.random_initial_date = random_initial_date
        self.stack_size = stack_size

        if reward_function not in list(self.reward_functions.keys()):
            raise Exception(
                f'Reward function not implemented. Please, choose between: {list(self.reward_functions.keys())}')

        self.reward_function = reward_function

        self.original_data = DataLoader(url).data

        if self.sampling_every is not None:
            new_dates = []
            new_prices = []
            for i in range(len(self.original_data['dates'])):
                if i % self.sampling_every == 0:
                    new_dates.append(self.original_data['dates'][i])
                    new_prices.append(self.original_data['prices'][i])
            self.original_data['dates'] = new_dates
            self.original_data['prices'] = np.asarray(new_prices)

        self.data = copy.deepcopy(self.original_data)

        self.reset()

    def reset(self):
        super(TradingGame, self).reset()

        if self.n_samples is not None:
            initial_position = np.random.randint(0, len(
                self.original_data['dates']) - self.n_samples) if self.random_initial_date else 1
            self.data['dates'] = self.original_data['dates'][-(self.n_samples + initial_position):-initial_position]
            self.data['prices'] = self.original_data['prices'][-(self.n_samples + initial_position):-initial_position]

        self.rewards = {'dates': [], 'rewards': []}

        self.stack = []
        self.current_day_index = 0

        self.buy_actions = {'dates': [], 'prices': []}
        self.sell_actions = {'dates': [], 'prices': []}

        # for the computation of the reward
        self.buy_signals = []  # contains 1 for buy actions and 0 otherwise
        self.sell_signals = []  # same but for sell actions
        self.P = []  # prices until now
        self.n = 0  # number of buy-sell pairs

    def step(self, action=None):
        """
            To perform a step:
                - updating current day;
                - checking if this is the last day;
        :param action: 0 (BUY) or 1 (SELL)
        :return:
        """
        # for th computation of the reward: Allen and Karjalainen's method
        self.update_reward_parameters(action)

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
            self.sell()
            return self.stack, done
        else:
            # otherwise we call a recursion until stack is full
            if len(self.stack) < self.stack_size:
                return self.step()
            else:
                # in case of normal situations, we return the updated stack
                return self.stack, done

    def update_reward_parameters(self, action):
        # updating prices list
        if action == 0 or action == 1:
            self.P.append(self.get_data_now()['price'])
        # updating the signals and n
        if action == 0:  # BUY
            performed = self.buy()
            if performed:
                self.buy_signals.append(1)  # adding the signal to the list
                self.sell_signals.append(0)
            else:
                # here only if the action was buy, but we have just bought
                self.buy_signals.append(0)
                self.sell_signals.append(0)
        elif action == 1:  # SELL
            performed = self.sell()
            if performed:
                self.buy_signals.append(0)
                self.sell_signals.append(1)
                self.n += 1  # in this case we have a complete pair
            else:
                self.buy_signals.append(0)
                self.sell_signals.append(0)

    def buy(self):
        data_now = self.get_data_now()
        performed = super(TradingGame, self).buy(data_now['price'])
        if performed:
            self.buy_actions['dates'].append(data_now['date'])
            self.buy_actions['prices'].append(data_now['price'])

        return performed

    def sell(self):
        data_now = self.get_data_now()
        performed = super(TradingGame, self).sell(data_now['price'])
        if performed:
            self.sell_actions['dates'].append(data_now['date'])
            self.sell_actions['prices'].append(data_now['price'])

        return performed

    def get_profit(self):
        data_now = self.get_data_now()
        return super(TradingGame, self).get_profit(data_now['price'])

    def get_reward(self):
        reward = self.reward_functions[self.reward_function](P=np.asarray(self.P),
                                                             I_b=np.asarray(self.buy_signals),
                                                             I_s=np.asarray(self.sell_signals),
                                                             n=self.n,
                                                             buy_fee=self.buy_fee, sell_fee=self.sell_fee)
        data_now = self.get_data_now()
        self.rewards['dates'].append(data_now['date'])
        self.rewards['rewards'].append(reward)
        return reward

    def get_data_now(self):
        return {'date': self.data['dates'][self.current_day_index],
                'price': self.data['prices'][self.current_day_index]}

    def plot_chart(self):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

        ax1.set_title(f'Total profit: {round(self.get_profit(), 2)} % (fee: {self.buy_fee} %)')

        ax1.plot(self.data['dates'], self.data['prices'], alpha=0.7, label='Price', zorder=1)

        ax1.axvline(self.data['dates'][self.stack_size], 0, 1, c='C2', alpha=0.3, label='Observation limit')

        ax1.scatter(self.buy_actions['dates'],
                    self.buy_actions['prices'],
                    marker='^', c='g', label='BUY', zorder=2)

        ax1.scatter(self.sell_actions['dates'],
                    self.sell_actions['prices'],
                    marker='v', c='r', label='SELL', zorder=2)

        ax1.set_ylabel(f'USD/{self.crypto_currency}')
        ax1.set_xlabel('Time')
        ax1.legend()

        ax2.set_title(f'{self.reward_function} Reward Function')
        ax2.plot(self.rewards['dates'], self.rewards['rewards'])
        ax2.plot(self.data['dates'], np.full((len(self.data['dates']),), np.average(self.rewards['rewards'])), alpha=0)
        ax2.set_ylabel(f'Reward')
        ax2.set_xlabel('Time')

        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

        interval = self.data['dates'][-1] - self.data['dates'][-2]
        initial_date = self.data['dates'][0]
        final_date = self.data['dates'][-1]

        plt.figtext(0.01,
                    0.01,
                    f'Sampling interval: {round(interval.total_seconds() / 60)} minutes\n'
                    f'Total number of days: {round((((final_date - initial_date).total_seconds() / 60) / 60) / 24)}\n'
                    f'Initial date: {initial_date} - Final date: {final_date}',
                    fontsize=20,
                    verticalalignment='bottom',
                    c='C2',
                    alpha=0.5)

        plt.show()
