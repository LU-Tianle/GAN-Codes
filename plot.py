import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]


def plot(name, value):
    _since_last_flush[name][_iter[0]] = value
    _iter[0] += 1


def flush(path):
    for name, vals in _since_last_flush.items():
        _since_beginning[name].update(vals)
        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]
        plt.figure(figsize=(11.5, 4.5))
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(path, name + '.jpg'))
        plt.close('all')
    _since_last_flush.clear()
