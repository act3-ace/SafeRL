import graphing_components
# this file reproduces paper plots

def paper_rejoin_plots(logdir):
    success_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','success_mean',int(1e5))
    reward_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','episode_reward_mean',int(1e5))
    eps_length_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','episode_length_mean',int(1e5))
    return success_plot,reward_plot,eps_length_plot

def paper_docking_plots(logdir):
    success_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','success_mean',int(1e5))
    reward_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','episode_reward_mean',int(1e5))
    eps_length_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','episode_length_mean',int(1e5))
    constr_viol_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','ratio_mean',int(1e5))
    delta_v_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','delta_v_mean',int(1e5))
    return success_plot,reward_plot,eps_length_plot,constr_viol_plot,delta_v_plot

if __name__ == '__main__':
    logdir =
