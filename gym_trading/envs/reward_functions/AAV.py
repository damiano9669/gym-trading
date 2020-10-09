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

    P_s = st.add_percentage(P, -sell_fee) * I_s
    P_b = st.add_percentage(P, buy_fee) * I_b

    if P_s[-1] != 0:
        P_b = P_b[np.where(P_b != 0)]
        try:
            return P_s[-1] - P_b[-1]
        except:
            return 0
    else:
        return 0

    # P_s = P_s[np.where(P_s != 0)]
    # P_b = P_b[np.where(P_b != 0)]

    # P_b = P_b[:-1] if P_b.shape[0] > P_s.shape[0] else P_b

    # incremental_aav = 0
    # for ni, (ps, pb) in enumerate(zip(P_s, P_b)):
    #     xi = ps - pb
    #     incremental_aav += (xi - incremental_aav) / (ni + 100)  # (ni +1) is for the real incremental mean
    # return incremental_aav
