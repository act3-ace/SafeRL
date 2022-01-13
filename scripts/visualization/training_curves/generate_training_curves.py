import os
import graphing_components
from tqdm import tqdm
# this file reproduces paper plots


def generate_training_curves(task, logdir, clip_method, output_dir='./', rename_map=None, **kwargs):
    assert task in ['docking', 'rejoin'], "task must be one of ['docking', 'rejoin']"

    q2_labels = ['success_mean', 'episode_reward_mean', 'episode_len_mean']
    if task == 'docking':
        q2_labels += ['ratio_mean', 'delta_v_total_mean']

    plots = []

    for q2_label in tqdm(q2_labels, position=1, leave=False):
        plots.append(graphing_components.graph_q1_v_q2(
            logdir, 'timesteps_total', q2_label, clip_method, output_dir=output_dir, rename_map=rename_map, **kwargs))

    return plots


if __name__ == '__main__':
    experiments = [
        (
            '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/rejoin_2d/rejoin_2d_fixed_nominal_20211014_203850/expr_20211014_203850', # noqa
            'figs/rejoin_2d',
            'rejoin',
            2e6,
        ),
        (
            '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/rejoin_3d_tall/expr_20211015_043032', # noqa
            'figs/rejoin_3d',
            'rejoin',
            3.5e6,
        ),
        (
            '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_2d/expr_20220106_192325',
            'figs/docking_2d',
            'docking',
            2e6,
        ),
        (
            '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_3d/expr_20220106_201704',
            'figs/docking_3d',
            'docking',
            4e6,
        ),
        (
            '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_oriented_2d/expr_20220102_224631',
            'figs/docking_oriented_2d',
            'docking',
            6e6,
        ),
    ]

    rename_map = {
        'timesteps_total': 'Timesteps',
        'success_mean': 'Success Mean',
        'episode_reward_mean': 'Average Return',
        'episode_len_mean': 'Episode Length',
        'ratio_mean': 'Constraint Violation Ratio',
        'delta_v_total_mean': 'Delta-v',
    }

    fig_width = 2.16667 +.16 #2.16667
    font_size = 8
    tick_font_size = font_size - 2

    rc_params = {
        'figure.figsize': (fig_width, fig_width),
        'figure.dpi': 300,
        'font.size': font_size,
        'xtick.major.pad': 0,
        'xtick.minor.pad': 0,
        'xtick.labelsize': tick_font_size,
        'ytick.major.pad': 0,
        'ytick.minor.pad': 0,
        'ytick.labelsize': tick_font_size,
        'lines.linewidth': 0.75,
        'lines.markersize': 2.5,
        'legend.fontsize': font_size,
        'legend.borderpad': 0.2,
        'legend.labelspacing': 0.3,
        'legend.markerscale': 20,
        'legend.handlelength': 1,
        'legend.handletextpad': 0.3,
        'axes.labelsize': font_size,
        'axes.labelpad': 1,
    }

    for log_dir, output_dir, task, clipping_bound in tqdm(experiments, position=0):
        os.makedirs(output_dir, exist_ok=True)
        generate_training_curves(task, log_dir, clipping_bound, output_dir=output_dir, rename_map=rename_map, interp_subsample_len=1000, rc_params=rc_params)
