import copy

import numpy as np

from gym_trading.envs.DataLoader import get_all_data
from gym_trading.envs.RenderStyle import *
from gym_trading.envs.Trader import Trader
from gym_trading.envs.reward_functions.AAV import get_AAV
from gym_trading.envs.reward_functions.AllenKarjalainen import get_AllenKarjalainen_excess_return
from gym_trading.envs.reward_functions.ROI import get_ROI


class TradingGame(Trader):
    reward_functions = {'ROI': get_ROI,
                        'AAV': get_AAV,
                        'Allen_and_Karjalainen': get_AllenKarjalainen_excess_return}

    def __init__(self,
                 n_samples=None,
                 sampling_every=None,
                 random_initial_date=False,
                 stack_size=1,
                 fee=0.25,
                 reward_function='AAV'):
        """

        :param n_samples: number of samples to load from the database
        :param stack_size: size of the observations
        :param fee:
        """
        super().__init__(init_amount=1.0,
                         init_currency='USD',
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

        self.original_data = get_all_data()

        if self.sampling_every is not None:
            new_dates = []
            new_prices = {key: [] for key in list(self.original_data.keys())[1:]}  # [1:] is to skip 'dates'
            for i in range(len(self.original_data['dates'])):
                # extract every n samples
                if i % self.sampling_every == 0:
                    new_dates.append(self.original_data['dates'][i])
                    for key in new_prices.keys():
                        new_prices[key].append(self.original_data[key][i])
            # set to original data
            self.original_data['dates'] = new_dates
            for key in new_prices.keys():
                self.original_data[key] = np.asarray(new_prices[key])

        self.data = copy.deepcopy(self.original_data)

        self.reset()

    def reset(self):
        super(TradingGame, self).reset()

        if self.n_samples is not None:
            initial_position = np.random.randint(0, len(
                self.original_data['dates']) - self.n_samples) if self.random_initial_date else 1
            self.data['dates'] = self.original_data['dates'][-(self.n_samples + initial_position):-initial_position]
            for key in list(self.data.keys())[1:]:
                self.data[key] = self.original_data[key][
                                 -(self.n_samples + initial_position):-initial_position]

        self.rewards = {'dates': [], 'rewards': []}

        self.stack = []
        self.current_day_index = 0

        # for the computation of the reward
        self.buy_signals = {key: [] for key in list(self.data.keys())[1:]}  # contains 1 for buy actions and 0 otherwise
        self.sell_signals = {key: [] for key in list(self.data.keys())[1:]}  # same but for sell actions
        self.P = {key: [] for key in list(self.data.keys())[1:]}  # prices until now
        self.n = {key: 0 for key in list(self.data.keys())[1:]}  # number of buy-sell pairs

    def step(self, action=None):
        """
            To perform a step:
                - updating current day;
                - checking if this is the last day;
        :param action: 0 (BUY_BTC),  1 (SELL_BTC), 2 (BUY_XRP), 3 (SELL_XRP), 4 (BUY_ETH), 5 (SELL_ETH)
        :return:
        """
        # for th computation of the reward
        self.update_reward_parameters(action)

        done = False
        self.current_day_index += 1
        if self.current_day_index >= len(self.data['dates']):
            self.current_day_index -= 1
            done = True

        data = self.get_data_now()
        # adding element to the list
        self.stack.append(data)
        # checking if the list exceeds the maximum number
        if len(self.stack) > self.stack_size:
            self.stack.pop(0)

        # if we have done we return the stack, also in case of non-full stack
        if done:
            # self.sell()
            return self.stack, done
        else:
            # otherwise we call a recursion until stack is full
            if len(self.stack) < self.stack_size:
                return self.step()
            else:
                # in case of normal situations, we return the updated stack
                return self.stack, done

    def update_reward_parameters(self, action):
        for key in list(self.data.keys())[1:]:

            # updating prices list
            if action in range(0, 6):
                self.P[key].append(self.get_data_now()[key.replace('s', '')])
            else:
                self.P[key].append(0)
            # updating the signals and n
            if (
                    (action == 0 and key == 'BTC_prices') or
                    (action == 2 and key == 'XRP_prices') or
                    (action == 4 and key == 'ETH_prices')):  # BUY
                performed = self.buy(key.replace('_prices', ''))
                if performed:
                    self.buy_signals[key].append(1)  # adding the signal to the list
                    self.sell_signals[key].append(0)
                else:
                    # here only if the action was buy, but we have just bought
                    self.buy_signals[key].append(0)
                    self.sell_signals[key].append(0)
            elif (
                    (action == 1 and key == 'BTC_prices') or
                    (action == 3 and key == 'XRP_prices') or
                    (action == 5 and key == 'ETH_prices')):  # SELL
                performed = self.sell(key.replace('_prices', ''))
                if performed:
                    self.buy_signals[key].append(0)
                    self.sell_signals[key].append(1)
                    self.n[key] += 1  # in this case we have a complete pair
                else:
                    self.buy_signals[key].append(0)
                    self.sell_signals[key].append(0)
            else:
                self.buy_signals[key].append(0)
                self.sell_signals[key].append(0)

    def buy(self, crypto_currency):
        key = f'{crypto_currency}_price'
        data_now = self.get_data_now()
        return super(TradingGame, self).buy(price=data_now[key],
                                            crypto_currency=crypto_currency)

    def sell(self, crypto_currency):
        key = f'{crypto_currency}_price'
        data_now = self.get_data_now()
        return super(TradingGame, self).sell(price=data_now[key],
                                             crypto_currency=crypto_currency)

    def get_profit(self):
        data_now = self.get_data_now()
        return super(TradingGame, self).get_profit(data_now[f'{self.current_currency}_price'])

    def get_reward(self):
        reward = 0
        for key in list(self.data.keys())[1:]:
            reward += self.reward_functions[self.reward_function](P=np.asarray(self.P[key]),
                                                                  I_b=np.asarray(self.buy_signals[key]),
                                                                  I_s=np.asarray(self.sell_signals[key]),
                                                                  n=self.n[key],
                                                                  buy_fee=self.buy_fee, sell_fee=self.sell_fee)
        data_now = self.get_data_now()
        self.rewards['dates'].append(data_now['date'])
        self.rewards['rewards'].append(reward)
        return reward

    def get_data_now(self):
        data_now = {}
        for key in self.data.keys():
            data_now[key.replace('s', '')] = self.data[key][self.current_day_index]
        return data_now

    def plot_chart(self):

        fig, axs = plt.subplots(len(list(self.data.keys())[1:]) + 1, 1, figsize=(15, 20))

        fig.suptitle(f'Total profit: {round(self.get_profit(), 2)} % (fee: {self.buy_fee} %)')

        for key, ax in zip(list(self.data.keys())[1:], axs[:-1]):
            ax.plot(self.data['dates'], self.data[key], alpha=0.7, label='Price', zorder=1)

            ax.axvline(self.data['dates'][self.stack_size], 0, 1, c='g', alpha=0.5, label='Start Date')
            ax.axvline(self.get_data_now()['date'], 0, 1, c='r', alpha=0.5, label='End Date')

            ax.scatter([date for i, date in enumerate(self.data['dates']) if self.buy_signals[key][i] == 1],
                       [date for i, date in enumerate(self.data[key]) if self.buy_signals[key][i] == 1],
                       marker='^', c='g', label='BUY', zorder=2)

            ax.scatter([date for i, date in enumerate(self.data['dates']) if self.sell_signals[key][i] == 1],
                       [date for i, date in enumerate(self.data[key]) if self.sell_signals[key][i] == 1],
                       marker='v', c='r', label='SELL', zorder=2)

            crypto = key.replace('_prices', '')
            ax.set_ylabel(f'USD/{crypto}')
            ax.set_xlabel('Time')
            ax.legend()

        axs[-1].set_title(f'{self.reward_function} Reward Function')
        axs[-1].plot(self.rewards['dates'], self.rewards['rewards'])
        axs[-1].plot(self.data['dates'], np.full((len(self.data['dates']),), np.average(self.rewards['rewards'])),
                     alpha=0)
        axs[-1].set_ylabel(f'Reward')
        axs[-1].set_xlabel('Time')

        for ax in axs:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

        interval = self.data['dates'][-1] - self.data['dates'][-2]
        initial_date = self.data['dates'][0]
        final_date = self.data['dates'][-1]

        sampling_interval_minutes = round(interval.total_seconds() / 60)
        tot_days = round((((final_date - initial_date).total_seconds() / 60) / 60) / 24)

        fig.text(0.01,
                 0.02,
                 f'Sampling interval: {sampling_interval_minutes} minutes '
                 f'({round(sampling_interval_minutes / 60, 2)} hours)\n'
                 f'Total number of days: {tot_days} ({round(tot_days / 365, 2)} years)\n'
                 f'Initial date: {initial_date} - Final date: {final_date}',
                 fontsize=20,
                 c='white',
                 ha="left", va='bottom', bbox=dict(facecolor='grey', alpha=1.))

        fig.show()
