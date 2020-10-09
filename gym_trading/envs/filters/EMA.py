import numpy as np


class EMA:

    def __init__(self, period, limit_queue=None):
        self.period = period
        self.limit_queue = limit_queue
        self.values = None

        self.alpha = float(2 / (self.period + 1))

    def update(self, new_value):
        # to limit memory
        if self.limit_queue is not None and self.values is not None:
            if len(self.values) > self.limit_queue:
                self.values.pop(0)

        if self.values is not None:
            ema_formula = float((1 - self.alpha) * self.values[-1] + self.alpha * new_value)
            self.values.append(ema_formula)
        else:
            # first value
            self.values = [new_value]

        return np.asarray(self.values)

    def reset(self):
        self.values = None


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.arange(0, 100, 0.1)

    y = np.sin(x) * np.exp(-x / 50) + np.random.normal(0, 0.1, size=x.shape[0])
    plt.plot(x, y, label='noisy function')

    emas = [EMA(5), EMA(10), EMA(20)]

    for observation in y:
        for ema in emas:
            ema.update(observation)

    for ema in emas:
        plt.plot(x, ema.values, label=f'EMA period: {ema.period}')
    plt.legend()
    plt.show()
