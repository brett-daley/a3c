import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import re
import os
import yaml
import numpy as np


RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'
COLORS = ('blue', 'green', 'red', 'black')
SHOW_ERROR = True


lambdas = [0.55, 0.75, 0.95, 1.0]
history_lens = [1, 4]
seeds = [0, 1, 2]

environments = [
    'breakout',
    'beam_rider',
    'centipede',
    'fishing_derby',
    'name_this_game',
    'pong',
    'qbert',
    'road_runner',
    'seaquest',
    'space_invaders',
]


def get_epochs(text):
    return [int(m.group(1)) for m in re.finditer('Epoch ([0-9]+)', text)]

def get_performance(text):
    return [float(m.group(1)) for m in re.finditer('Mean reward (-?[0-9]+\.[0-9]+)', text)]

def get_stdev(text):
    return [float(m.group(1)) for m in re.finditer('Standard dev (-?[0-9]+\.[0-9]+)', text)]

def extract_axes_from_file(file):
    with open(file, 'r') as f:
        text = f.read()
        x_axis = get_epochs(text)
        y_axis = get_performance(text)
        stdev  = get_stdev(text)
    return np.array(x_axis), np.array(y_axis), np.array(stdev)


def make_title(env, len):
    env = env.replace('_', ' ').title()
    return '{} ({}-frame input)'.format(env, len)


def exists(files):
    exists = True
    for f in files:
        if not os.path.exists(f):
            print('Warning: could not find {}'.format(f))
            exists = False
    return exists


def create_plot(env, length):
    basename = 'a3c_{}_len{}_lve*_seed*.txt'.format(env, str(length))
    print('Starting target \'{}\''.format(basename))

    title = make_title(env, length)
    files = [basename.replace('*', str(lve), 1) for lve in lambdas]
    files = [os.path.join(RESULTS_DIR, f) for f in files]

    legend_title = 'A3C($\\lambda$)'
    legend = ['$\\lambda={}$'.format(lve) for lve in lambdas]
    xlabel = 'Epoch'
    ylabel = 'Score'

    # Make sure files exist for all seeds
    if not exists([f.replace('*', str(s)) for s in seeds for f in files]):
        print('Files were missing... skipped')
        return False

    # Now collect and average axes for each seed
    axis_dict = {}
    for f in files:
        for s in seeds:
            path = f.replace('*', str(s))
            x_axis, y_axis, _ = extract_axes_from_file(path)

            if f not in axis_dict.keys():
                axis_dict[f] = {}
                axis_dict[f]['x'] = x_axis
                axis_dict[f]['y'] = y_axis
                axis_dict[f]['stderr'] = np.square(y_axis)
            else:
                try:
                    assert (axis_dict[f]['x'].shape == x_axis.shape)
                    assert (axis_dict[f]['x'] == x_axis).all()
                    assert (axis_dict[f]['y'].shape == y_axis.shape)
                except AssertionError:
                    raise RuntimeError('{} has inconsistent format'.format(f))
                axis_dict[f]['y'] += y_axis
                axis_dict[f]['stderr'] += np.square(y_axis)

    N = len(seeds)
    for f, axes in axis_dict.items():
        axis_dict[f]['y'] /= N
        axis_dict[f]['stderr'] /= N
        axis_dict[f]['stderr'] -= np.square(axis_dict[f]['y'])
        axis_dict[f]['stderr'] = np.sqrt(axis_dict[f]['stderr']) / np.sqrt(N)

    # Generate the plot
    assert len(files) == len(legend)

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.figure()

    ylim = [float('inf'), -float('inf')]

    for i, f in enumerate(files):
        x_axis = axis_dict[f]['x']
        y_axis = axis_dict[f]['y']
        stderr = axis_dict[f]['stderr']
        plt.plot(x_axis, y_axis, COLORS[i], label=legend[i])
        if SHOW_ERROR:
            plt.fill_between(x_axis, (y_axis - stderr), (y_axis + stderr), color=COLORS[i], alpha=0.25, linewidth=0)
        n_epochs = len(x_axis) - 1

        ylim[0] = min(ylim[0], min(y_axis))
        ylim[1] = max(ylim[1], max(y_axis))

    ylim[0] = (0.95 * ylim[0]) if (ylim[0] > 0.0) else (1.05 * ylim[0])
    ylim[1] = (1.05 * ylim[1]) if (ylim[1] > 0.0) else (0.95 * ylim[1])

    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=17.5)
    plt.ylabel(ylabel, fontsize=17.5)
    legend = plt.legend(loc='best', framealpha=0.5, fontsize=16)
    legend.set_title(legend_title, prop={'size': 16, 'weight': 'heavy'})

    plt.xlim([0, n_epochs])
    plt.ylim(ylim)

    plt.tight_layout(pad=0)
    fig = plt.gcf()

    output_name = basename.rstrip('lve*_seed*.txt') + '.png'
    fig.savefig(os.path.join(PLOTS_DIR, output_name))
    print('Plot saved as {}'.format(output_name))

    return True


def main():
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)

    n_success = 0
    n_total = 0
    for env in environments:
        for len in history_lens:
            n_total += 1
            if create_plot(env, len):
                n_success += 1
            print(flush=True)
    print('Generated {}/{} plots.'.format(n_success, n_total))


if __name__ == '__main__':
    main()
