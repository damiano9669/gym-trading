import my_utils.math.stats as st


def get_ROI(P, I_b, I_s, n, buy_fee=0.25, sell_fee=0.25):
    """
        Excess return or fitness for trading
    :param P: prices collected
    :param I_b: buy signals collected
    :param I_s: sell signals collected
    :param n: number of buy-sell pairs
    :param buy_fee: buy fee
    :param sell_fee: sell fee
    :return:
    """
    return (st.add_percentage(P, -sell_fee).T.dot(I_s) -
            st.add_percentage(P, buy_fee).T.dot(I_b)) / st.add_percentage(P, buy_fee).T.dot(I_b)
