import os
import graphing_components
# this file reproduces paper plots


def paper_rejoin_plots(logdir, clip_method, output_dir='./', rename_map=None):
    success_plot = graphing_components.graph_q1_v_q2(logdir, 'timesteps_total', 'success_mean', clip_method, output_dir=output_dir, rename_map=rename_map)
    reward_plot = graphing_components.graph_q1_v_q2(logdir, 'timesteps_total', 'episode_reward_mean', clip_method, output_dir=output_dir, rename_map=rename_map)
    eps_length_plot = graphing_components.graph_q1_v_q2(logdir, 'timesteps_total', 'episode_len_mean', clip_method, output_dir=output_dir, rename_map=rename_map)
    return success_plot, reward_plot, eps_length_plot


def paper_docking_plots(logdir, clip_method, output_dir='./', rename_map=None):
    success_plot = graphing_components.graph_q1_v_q2(logdir, 'timesteps_total', 'success_mean', clip_method, output_dir=output_dir, rename_map=rename_map)
    reward_plot = graphing_components.graph_q1_v_q2(logdir, 'timesteps_total', 'episode_reward_mean', clip_method, output_dir=output_dir, rename_map=rename_map)
    eps_length_plot = graphing_components.graph_q1_v_q2(logdir, 'timesteps_total', 'episode_len_mean', clip_method, output_dir=output_dir, rename_map=rename_map)
    constr_viol_plot = graphing_components.graph_q1_v_q2(logdir, 'timesteps_total', 'ratio_mean', clip_method, output_dir=output_dir, rename_map=rename_map)
    delta_v_plot = graphing_components.graph_q1_v_q2(logdir, 'timesteps_total', 'delta_v_total_mean', clip_method, output_dir=output_dir, rename_map=rename_map)
    return success_plot, reward_plot, eps_length_plot, constr_viol_plot, delta_v_plot


if __name__ == '__main__':
    log_dirs = [
        '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/rejoin_2d/rejoin_2d_fixed_nominal_20211014_203850/expr_20211014_203850',
        '/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_2d/expr_20211014_172912',
    ]

    output_dirs = [
        'rejoin_2d',
        'docking_2d',
        # 'rejoin_3d',
        # 'docking_3d',
    ]

    plot_fns = [
        paper_rejoin_plots,
        paper_docking_plots,
    ]

    clipping_bounds = [
        int(1e6),
        int(1.2e6),
    ]

    rename_map = {
        'timesteps_total': 'Timesteps',
        'success_mean': 'Success Mean',
        'episode_reward_mean': 'Average Return',
        'episode_len_mean': 'Episode Length',
        'ratio_mean': 'Constraint Violation Ratio',
        'delta_v_total_mean': 'Delta-v',
    }

    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    for i in range(len(log_dirs)):
        plot_fns[i](log_dirs[i], clipping_bounds[i], output_dir=output_dirs[i], rename_map=rename_map)
