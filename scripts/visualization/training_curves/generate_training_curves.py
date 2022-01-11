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
        # (
        #     '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/rejoin_2d/rejoin_2d_fixed_nominal_20211014_203850/expr_20211014_203850', # noqa
        #     'rejoin_2d',
        #     'rejoin',
        #     'shortest',
        # ),
        # (
        #     '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/rejoin_3d_tall/expr_20211015_043032', # noqa
        #     'rejoin_3d',
        #     'rejoin',
        #     'shortest',
        # ),
        (
            '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_2d/expr_20220106_192325',
            'docking_2d',
            'docking',
            'shortest',
        ),
        # (
        #     '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_3d/expr_20220106_201704',
        #     'docking_3d',
        #     'docking',
        #     'shortest',
        # ),
        # (
        #     '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_oriented_2d/expr_20220102_224631',
        #     'docking_oriented_2d',
        #     'docking',
        #     'shortest',
        # ),
    ]

    rename_map = {
        'timesteps_total': 'Timesteps',
        'success_mean': 'Success Mean',
        'episode_reward_mean': 'Average Return',
        'episode_len_mean': 'Episode Length',
        'ratio_mean': 'Constraint Violation Ratio',
        'delta_v_total_mean': 'Delta-v',
    }

    rc_params = {
        'figure.figsize': (0.65*3.375, 0.65*3.375*4.8/6.4),
        'figure.dpi': 300,
        'font.size': 10,
        'xtick.major.pad': 0,
        'xtick.minor.pad': 0,
        'xtick.labelsize': 10,
        'ytick.major.pad': 0,
        'ytick.minor.pad': 0,
        'ytick.labelsize': 10,
        'lines.linewidth': 0.75,
        'lines.markersize': 2.5,
        'legend.fontsize': 10,
        'legend.borderpad': 0.2,
        'legend.labelspacing': 0.3,
        'legend.markerscale': 20,
        'legend.handlelength': 1,
        'legend.handletextpad': 0.3,
        'axes.labelsize': 10,
    }

    for log_dir, output_dir, task, clipping_bound in tqdm(experiments, position=0):
        os.makedirs(output_dir, exist_ok=True)
        generate_training_curves(task, log_dir, clipping_bound, output_dir=output_dir, rename_map=rename_map, interp_subsample_len=1000, rc_params=rc_params)
