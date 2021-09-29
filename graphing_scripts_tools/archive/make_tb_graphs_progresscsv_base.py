import argparse
from ray.tune import Analysis
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='logging capacities')

    # should only be one
    parser.add_argument(
        '--logdir',
        type=str,
        default='.',
        help='Experiment directory to evaluate')


    args = parser.parse_args()

    return args


def get_succ_rate(expr_path):
    """
    The percent of cases that the control system successfully performs a mission.

    From: Safe Autononmy Metrics by Dr. Kerianne Hobbs
    """
    data = pd.read_csv(expr_path + '/progress.csv')
    if len(data.columns) > 70: # some do not have custom metrics available
        success_mean_index = [i for i, j in enumerate(data.columns)
                           if "success_mean" in j][0]
        return data.iloc[-1, success_mean_index]*100
    else:
        return float('nan')

def get_success_mean_col(data):
    return data['custom_metrics/outcome/success_mean']



def get_ep_length(expr_path):
    """
    The average time in seconds (or timesteps) required to successfully complete the
    task (an episode).

    From: Safe Autonomy Metrics by Dr. Kerianne Hobbs
    """
    data = pd.read_csv(expr_path + '/progress.csv')
    return data['episode_len_mean'].iloc[-1]

def main():
    #analysis = Analysis(experiment_dir=args.logdir, default_metric=args.metric, default_mode=args.mode)
    #df = analysis.dataframe()
    #print(df.columns)
    #print(df)
    args = get_args()
    #success_rate = get_succ_rate(args.logdir)

    data = pd.read_csv(args.logdir + '/progress.csv')

    success_mean_data = get_success_mean_col(data)
    plt.figure()
    plt.plot(success_mean_data)


if __name__ == '__main__':
    main()
