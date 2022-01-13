python plot_vel_constraint.py \
    --dir /data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_2d/expr_20220106_192444 \
    --checkpoints 2 9 40 100 \
    --only_plot

python plot_trajectory.py \
    --dir /data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_2d/expr_20220106_192444 \
    --checkpoints 2 9 40 100 \
    --alt_env_config ../../configs/docking/docking_pretty.yaml \
    --only_plot

python plot_trajectory.py \
    --dir /data/petabyte/safe_autonomy/ieee_aero_2022/experiments/rejoin_2d/rejoin_2d_fixed_nominal_20211014_203850/freq_ckpt_seed_913840577 \
    --task rejoin \
    --checkpoints 4 5 9 19 \
    --alt_env_config ../../configs/rejoin/rejoin_continuous_pretty.yaml \
    --only_plot
