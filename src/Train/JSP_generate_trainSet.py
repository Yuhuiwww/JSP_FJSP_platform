import os

import numpy as np
from JSP_config import get_config
config=get_config()
def permute_rows(x):
    """
    :param x: np array 2-D
    :return:
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    return times, machines


def override(fn):
    """
    override decorator
    """
    return fn



if __name__ == '__main__':
    size = 100 #Number of examples
    folder_path = 'JSP_learning_train'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    instances = np.array([uni_instance_gen(n_j=config.Pn_j, n_m=config.Pn_m, low=config.low, high=config.high) for _ in range(size)])
    np.save('JSP_learning_train/{}x{}.npy'.format(config.Pn_j, config.Pn_m), instances)