import numpy as np


def lerp(key1, key2, t1, t2):
    interval = t2 - t1
    linear_step = (key2 - key1) / interval
    linear_interp = np.expand_dims(key1, 1).repeat(interval + 1, axis=1) + \
                    np.expand_dims(linear_step, 1).repeat(interval + 1, axis=1) * np.arange(interval + 1)

    return linear_interp
