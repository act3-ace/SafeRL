import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_args():
    parser = argeparse.ArgumentParser()

    parser.add_argument('--exper_dir',
                        type=str,
                        help='Experiment directory to evaluate should start with PPO')

    parser.add_argument('--output_dir',
                        type=str,
                        default='../graphs_output/')

def plot_success_mean(data):
    steps = data['timesteps_total']
    success_mean= data['custom_metrics/outcome/success_mean']
    exper_name = args.logdir.split('/')[-2]
    graph_name = 'success_mean'
    save_to = exper_name + 'success_mean_graph.png'

    plt.figure()
    plt.title('Steps vs Success Mean')
    plt.ylabel('Success Mean')
    plt.xlabel('Number of steps')
    plt.plot(steps,success_mean)
    plt.save(save_to)

def plot_episode_len_mean(data,args):
    graph_name = 'episode_len_mean'
    steps = data['timesteps_total']
    episodes_len_mean= data['episode_len_mean']
    exper_name = args.logdir.split('/')[-2]
    save_to = exper_name + graph_name + .png

    plt.figure()
    plt.title('Steps vs Episode Mean length')
    plt.ylabel('Episode Mean Length')
    plt.xlabel('Number of steps')
    plt.plot(steps,episodes_len_mean)
    plt.save(save_to)

def main(args):
    data = pd.read_csv(args.logdir + args.exper_dir)
    plot_success_mean(data,args)
    plot_episode_len_mean(data,args)


if __name__ == "__main__":
    parsed_args = get_args()
    main(parsed_args)
