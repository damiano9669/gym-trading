import numpy as np

"""
    Source: https://www.cs.montana.edu/courses/spring2007/536/materials/Lopez/genetic.pdf
"""


def get_excess_return(P, I_b, I_s, n, buy_fee=0.25, sell_fee=0.25):
    """
        Excess return or fitness for trading
    :param P: prices collected
    :param I_b: buy signals collected
    :param I_s: sell signals collectes
    :param n: number of buy-sell pairs
    :param buy_fee: buy fee
    :param sell_fee: sell fee
    :return:
    """
    return get_r(P, I_b, I_s, n, buy_fee=buy_fee, sell_fee=sell_fee) - \
           get_r_bh(P, buy_fee=buy_fee, sell_fee=sell_fee)


def get_r(P, I_b, I_s, n, buy_fee=0.25, sell_fee=0.25):
    """
        The continuously compounded return for a trading rule
    :param P:
    :param I_b:
    :param I_s:
    :param n:
    :param buy_fee:
    :param sell_fee:
    :return:
    """
    return get_r_t(P).T.dot(I_b[1:]) + \
           get_r_f(P).T.dot(I_s[1:]) + \
           n * np.log((1 - buy_fee) / (1 + sell_fee))


def get_r_bh(P, buy_fee=0.25, sell_fee=0.25):
    """
         The return for the buy-and-hold strategy (buy the first day, sell the last day)
    :param P:
    :param buy_fee:
    :param sell_fee:
    :return:
    """
    return np.sum(get_r_t(P)) + np.log((1 - buy_fee) / (1 + sell_fee))


def get_r_t(P):
    """
         Daily continuously compounded return
    :param P:
    :return:
    """
    return np.log(P[1:]) - np.log(P[:-1])


def get_r_f(P):
    """
        The risk-free rate on day
    :param P:
    :return:
    """
    # Sincerely I don't know if this is correct
    x = np.arange(0, P.shape[0] - 1, 1)
    return np.exp(-x)
