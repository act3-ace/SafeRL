python plot_vel_constraint.py \
    --dir /data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_2d/expr_20220106_192444 \
    --checkpoints 3 9 40 100

python plot_trajectory.py \
    --dir /data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_2d/expr_20220106_192444 \
    --checkpoints 3 9 40 100 \
    --alt_env_config ../../configs/docking/docking_pretty.yaml