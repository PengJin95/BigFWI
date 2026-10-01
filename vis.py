import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Load colormap for velocity map visualization
rainbow_cmap = ListedColormap(np.load('rainbow256.npy'))

def plot_velocity(output, target, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    if vmin is None or vmax is None:
        vmax, vmin = np.max(target), np.min(target)
    im = ax[0].matshow(output, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.08)
    ax[1].matshow(target, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.08)
    fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
    plt.savefig(path)
    plt.close('all')


def plot_seismic(output, target, path, vmin=-1e-5, vmax=1e-5):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    aspect = output.shape[1]/output.shape[0]
    im = ax[0].matshow(target, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title('Ground Truth')
    ax[1].matshow(output, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title('Prediction')
    ax[2].matshow(output - target, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].set_title('Difference')

    fig.colorbar(im, ax=ax, shrink=0.75, label='Amplitude')
    plt.savefig(path)
    plt.close('all')

