import graphing_components
# this file reproduces paper plots

def paper_rejoin_plots(logdir,clip_method):
    #success_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','success_mean',clip_method)
    #reward_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','episode_reward_mean',clip_method)
    eps_length_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','episode_len_mean',clip_method)
    return success_plot,reward_plot,eps_length_plot

def paper_docking_plots(logdir,clip_method):
    success_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','success_mean',clip_method)
    reward_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','episode_reward_mean',clip_method)
    eps_length_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','episode_length_mean',clip_method)
    constr_viol_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','ratio_mean',clip_method)
    delta_v_plot = graphing_components.graph_q1_v_q2(logdir,'timesteps_total','delta_v_mean',clip_method)
    return success_plot,reward_plot,eps_length_plot,constr_viol_plot,delta_v_plot

if __name__ == '__main__':
    #docking_logdir = '/home/vgangal/Downloads/expr_20210914_005226'
    rejoin_logdir = '/home/vgangal/Downloads/expr_20210909_194211'
    paper_rejoin_plots(rejoin_logdir,int(5.5e5))
    #paper_docking_plots(docking_logdir,int(1e5))
