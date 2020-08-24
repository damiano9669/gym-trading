import my_utils.math.stats as st

import numpy as np


def get_AAV(P, I_b, I_s, n, buy_fee=0.25, sell_fee=0.25):
    """
        Fitness for trading
    :param P: prices collected
    :param I_b: buy signals collected
    :param I_s: sell signals collected
    :param n: number of buy-sell pairs
    :param buy_fee: buy fee
    :param sell_fee: sell fee
    :return:
    """
    if n == 0:
        return 0

    P_s_prime = st.add_percentage(P, -sell_fee) * I_s
    P_b_prime = st.add_percentage(P, buy_fee) * I_b

    P_s = []
    P_b = []
    for ps, pb in zip(P_s_prime, P_b_prime):
        if ps != 0:
            P_s.append(ps)
        if pb != 0:
            P_b.append(pb)

    P_b = P_b[:-1] if len(P_b) > len(P_s) else P_b

    return np.sum(np.asarray(P_s) - np.asarray(P_b)) / n
