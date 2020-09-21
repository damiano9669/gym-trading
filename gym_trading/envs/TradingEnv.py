import gym
import numpy as np
from gym.spaces import Discrete

from gym_trading.envs.TradingGame import TradingGame


class TradingEnv(gym.Env):

    def __init__(self,
                 n_samples=None,
                 sampling_every=None,
                 random_initial_date=False,
                 stack_size=1,
                 fee=0.25,
                 reward_function='AAV',
                 endurance_mode=False,
                 normalize_observation=False,
                 gan_generation=False,
                 new_generation_onreset=True):
        """

        :param n_samples: number of total samples
        :param sampling_every: interval time between two samples.
                If 1, the interval is 15 minutes, 2 -> 30 minutes and so on
        :param random_initial_date: the samples can be taken from a randomly starting date
        :param stack_size: size of the stack of the past observations
        :param fee: fee for conversion
        :param reward_function: reward function to choose between ROI, AAV and Allen_and_Karjalainen
        :param endurance_mode: if the agent performs a bad action it dies and we stop the game
        :param normalize_observation: to normalize the observations
        :param gan_generation: I trained a neural network being able to generate time series like BTC prices.
                The model has been trained from the BTC historical data of the last 3 years
        :param new_generation_onreset: If the ga_generation is enable,
                we can generate new data at each reset of the environment
        """
        self.endurance_mode = endurance_mode
        self.normalize = normalize_observation
        self.action_space = Discrete(6)
        # 0 (BUY_BTC),  1 (SELL_BTC),
        # 2 (BUY_XRP), 3 (SELL_XRP),
        # 4 (BUY_ETH), 5 (SELL_ETH)
        self.trader = TradingGame(n_samples=n_samples,
                                  sampling_every=sampling_every,
                                  random_initial_date=random_initial_date,
                                  stack_size=stack_size,
                                  fee=fee,
                                  reward_function=reward_function,
                                  gan_generation=gan_generation,
                                  new_generation_onreset=new_generation_onreset)
        ob = self.reset()
        self.observation_space = np.zeros(shape=ob.shape)

    def step(self, action):
        """

        :param action: 0 (BUY_BTC),  1 (SELL_BTC), 2 (BUY_XRP), 3 (SELL_XRP), 4 (BUY_ETH), 5 (SELL_ETH)
        :return: observation, reward, done, infos -> observation can be an unique price or a numpy array of prices.
        """

        observation, done = self.trader.step(action)
        observation = self.clean_observation(observation)

        reward = self.trader.get_reward()
        if len(self.trader.rewards['rewards']) > 2:
            if self.endurance_mode and self.trader.rewards['rewards'][-1] < self.trader.rewards['rewards'][-2]:
                done = True  # stopping condition in case of loosing money

        return observation, reward, done, {}

    def reset(self):
        self.trader.reset()
        observation, done = self.trader.step()
        observation = self.clean_observation(observation)
        return observation

    def render(self):
        self.trader.plot_chart()

    def get_profit(self):
        return self.trader.get_profit()

    def clean_observation(self, observation):
        crypto_obs = []
        for key in list(self.trader.data.keys())[1:]:
            crypto_stack = []
            for ob in observation:
                crypto_stack.append(ob[key.replace('s', '')])
            crypto_obs.append(np.asarray(crypto_stack))

        if self.normalize:
            for i, crypto_ob in enumerate(crypto_obs):
                crypto_obs[i] = self.normalize_observation(crypto_ob)
                # Is the line below necessary?
                crypto_obs[i] = crypto_obs[i] / (np.max(crypto_obs[i] + 1e-20))

        # observation = crypto_obs[0] * crypto_obs[1] * crypto_obs[2]
        observation = np.stack(crypto_obs, axis=-1)

        return observation

    def normalize_observation(self, x):
        x_mean = np.mean(x)
        x_std = np.std(x)
        return (x - x_mean) / (x_std + 1e-20)
