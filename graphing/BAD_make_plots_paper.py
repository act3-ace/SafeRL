import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir',
                        type=str,
                        help = 'top level directory containing all experiment dirs')
                        
    parser.add_argument('--experdir',
                        type=str,
                        help='Experiment directory to evaluate should start with PPO')

    parser.add_argument('--outputdir',
                        type=str,
                        default='../graphs_output/')
    
    return parser.parse_args()

def plot_success_mean(data,args):
    steps = data['timesteps_total']
    success_mean= data['custom_metrics/outcome/success_mean']
    exper_name = args.experdir.split('/')[-2]
    graph_name = 'success_mean'
    save_to = exper_name + 'success_mean_graph.png'

    plt.figure()
    plt.title('Steps vs Success Mean')
    plt.ylabel('Success Mean')
    plt.xlabel('Number of steps')
    plt.plot(steps,success_mean)
    plt.savefig(save_to)

def plot_episode_len_mean(data,args):
    graph_name = 'episode_len_mean'
    steps = data['timesteps_total']
    episodes_len_mean= data['episode_len_mean']
    exper_name = args.experdir.split('/')[-2]
    save_to = exper_name + graph_name + '.png'

    plt.figure()
    plt.title('Steps vs Episode Mean length')
    plt.ylabel('Episode Mean Length')
    plt.xlabel('Number of steps')
    plt.plot(steps,episodes_len_mean)
    plt.savefig(save_to)

def plot_multiple_success_means(args):     
    logdir = args.logdir 
    all_subdirs = next(os.walk(logdir))[1]
    plt.figure()
    plt.title('Steps vs Success Mean')
    plt.ylabel('Success Mean')
    plt.xlabel('Number of steps')

    seed = 0
    for d in all_subdirs: 
        if d == 'training_logs':
            continue
        else: 
            seed_name = 'seed_' + str(seed)
            csv_path = logdir + '/' + d + '/' + 'progress.csv'
            data = pd.read_csv(csv_path)
            steps = data['timesteps_total']
            success_mean= data['custom_metrics/outcome/success_mean']
            plt.plot(steps,success_mean,label=seed_name)
            seed = seed + 1
            #print(seed)
    
    plt.legend()
    
    exper_name = logdir.split('/')[-2]
    env_name = d.split('_')[1]
    graph_name = 'multi_seed_success_mean_graph'
    ext = '.png'
    output_file = exper_name + env_name + graph_name + ext
    plt.savefig(output_file)
    

def plot_multiple_eps_len_mean(args):
    logdir = args.logdir 
    all_subdirs = next(os.walk(logdir))[1]
    plt.figure()
    plt.title('Steps vs Episode Mean Length')
    plt.ylabel('Episode Mean Length')
    plt.xlabel('Number of steps')

    seed = 0
    for d in all_subdirs: 
        if d == 'training_logs':
            continue
        else: 
            seed_name = 'seed_' + str(seed)
            csv_path = logdir + '/' + d + '/' + 'progress.csv'
            data = pd.read_csv(csv_path)
            steps = data['timesteps_total']
            episodes_len_mean= data['episode_len_mean']
            plt.plot(steps,episodes_len_mean,label=seed_name)
            seed = seed + 1
            #print(seed)
    
    plt.legend()
    
    exper_name = logdir.split('/')[-2]
    env_name = d.split('_')[1]
    graph_name = 'multi_seed_eps_len_mean_graph'
    ext = '.png'
    output_file = exper_name + env_name + graph_name + ext
    plt.savefig(output_file)
    
    
def main(args):
    plot_multiple_success_means(args)
    plot_multiple_eps_len_mean(args)
    

if __name__ == "__main__":
    parsed_args = get_args()
    main(parsed_args)
