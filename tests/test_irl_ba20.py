import sys
from pathlib import Path
p = Path(__file__).absolute().parent.parent
if sys.path[-1] != str(p):
    sys.path.append(str(p))
print(str(p))

from graph_irl.buffer_v2 import GraphBuffer
from graph_irl.policy import GaussPolicy, TwoStageGaussPolicy, TanhGaussPolicy, GCN, Qfunc
from graph_irl.graph_rl_utils import *
from graph_irl.transforms import *
from graph_irl.sac import SACAgentGraph, TEST_OUTPUTS_PATH
from graph_irl.reward import GraphReward, StateGraphReward
from graph_irl.examples.circle_graph import create_circle_graph
from graph_irl.irl_trainer import IRLGraphTrainer
from graph_irl.eval_metrics import save_graph_stats_k_runs_GO1, save_analysis_graph_stats

from functools import partial
import re

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import barabasi_albert_graph

import random
import numpy as np
import torch


def trig_circle_init(*args):
    n_nodes = args[0]
    inputs = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    inputs = torch.tensor(inputs, dtype=torch.float32).view(-1, 1)
    return torch.cat((torch.cos(inputs), torch.sin(inputs)), -1)


n_nodes = 20
expert_edge_index = get_consec_edge_index(barabasi_albert_graph(n_nodes, 3))
nodes = trig_circle_init(n_nodes)
ba20 = Data(x=nodes, edge_index=expert_edge_index)

# circular graph with 7 nodes;
n_nodes, node_dim = ba20.x.shape
num_edges_expert = ba20.edge_index.shape[-1] // 2

# graph transform setup;
transform_fn_ = None
n_cols_append = 0  # based on transform_fn_

get_graph_level_feats_fn = None
n_extra_cols_append = 0  # based on get_graph_level_feats_fn

# instantiate transform;
transform_ = None

print(f"IRL training for {n_nodes}-node graph")

def get_params(
    net_hiddens=64,
    encoder_h=[8, 64],
    embed_dim=8,
    bet_on_homophily=False,
    net2_batch_norm=False,
    with_batch_norm=False,
    final_tanh=True,
    action_is_index=True,
    do_dfs_expert_paths=True,
    UT_trick=False,
    per_decision_imp_sample=True,
    weight_scaling_type='abs_max',
    n_nodes=n_nodes,
    node_dim=node_dim,
    nodes=nodes,
    n_cols_append=n_cols_append,
    n_extra_cols_append=n_extra_cols_append,
):
    print(n_nodes, node_dim, nodes.shape)
    # some setup;
    encoder_hiddens = [encoder_h, encoder_h, embed_dim] if isinstance(encoder_h, int) else encoder_h + [embed_dim]
    reward_fn_hiddens = [net_hiddens, net_hiddens]
    gauss_policy_hiddens = [net_hiddens, net_hiddens]
    tsg_policy_hiddens1 = [net_hiddens, net_hiddens]
    tsg_policy_hiddens2 = [net_hiddens]
    qfunc_hiddens = [net_hiddens, net_hiddens]
    which_reward_fn = 'state_reward_fn'
    which_policy_kwargs = 'tanh_gauss_policy_kwargs'
    action_dim = embed_dim * 2
    reward_scale = embed_dim

    encoder_dict = dict(
        encoder = GCN(node_dim + n_cols_append, encoder_hiddens, 
                      with_batch_norm=with_batch_norm, 
                      final_tanh=final_tanh,
                      bet_on_homophily=bet_on_homophily, 
                      net2_batch_norm=net2_batch_norm),
        encoderq1 = GCN(node_dim + n_cols_append, encoder_hiddens, 
                      with_batch_norm=with_batch_norm, 
                      final_tanh=final_tanh,
                      bet_on_homophily=bet_on_homophily, 
                      net2_batch_norm=net2_batch_norm), 
        encoderq2 = GCN(node_dim + n_cols_append, encoder_hiddens, 
                      with_batch_norm=with_batch_norm, 
                      final_tanh=final_tanh,
                      bet_on_homophily=bet_on_homophily, 
                      net2_batch_norm=net2_batch_norm),
        encoderq1t = GCN(node_dim + n_cols_append, encoder_hiddens, 
                      with_batch_norm=with_batch_norm, 
                      final_tanh=final_tanh,
                      bet_on_homophily=bet_on_homophily, 
                      net2_batch_norm=net2_batch_norm),
        encoderq2t = GCN(node_dim + n_cols_append, encoder_hiddens, 
                      with_batch_norm=with_batch_norm, 
                      final_tanh=final_tanh,
                      bet_on_homophily=bet_on_homophily, 
                      net2_batch_norm=net2_batch_norm),
        encoder_reward = GCN(node_dim + n_cols_append, encoder_hiddens, 
                      with_batch_norm=with_batch_norm, 
                      final_tanh=final_tanh,
                      bet_on_homophily=bet_on_homophily, 
                      net2_batch_norm=net2_batch_norm),
    )

    reward_funcs = dict(
        reward_fn = GraphReward(
            encoder_dict['encoder_reward'], 
            embed_dim=encoder_hiddens[-1] + n_extra_cols_append, 
            hiddens=reward_fn_hiddens, 
            with_layer_norm=True,
            with_batch_norm=False,
        ),
        state_reward_fn=StateGraphReward(
            encoder_dict['encoder_reward'], 
            embed_dim=encoder_hiddens[-1] + n_extra_cols_append, 
            hiddens=reward_fn_hiddens, 
            with_layer_norm=True,
            with_batch_norm=False,
        )
    )

    reward_fn = reward_funcs[which_reward_fn]

    policy_constructors = dict(
        tsg_policy_kwargs=TwoStageGaussPolicy,
        gauss_policy_kwargs=GaussPolicy,
        tanh_gauss_policy_kwargs=TanhGaussPolicy
    )
    
    policy_kwargs = dict(
        gauss_policy_kwargs = dict(
            obs_dim=encoder_hiddens[-1] + n_extra_cols_append,
            action_dim=encoder_hiddens[-1],
            hiddens=gauss_policy_hiddens,
            with_layer_norm=True,
            encoder=encoder_dict['encoder'],
            two_action_vectors=True,
        ),
        tanh_gauss_policy_kwargs = dict(
            obs_dim=encoder_hiddens[-1] + n_extra_cols_append,
            action_dim=encoder_hiddens[-1],
            hiddens=gauss_policy_hiddens,
            with_layer_norm=True,
            encoder=encoder_dict['encoder'],
            two_action_vectors=True,
        ),
        tsg_policy_kwargs = dict(
            obs_dim=encoder_hiddens[-1] + n_extra_cols_append,
            action_dim=encoder_hiddens[-1],
            hiddens1=tsg_policy_hiddens1,
            hiddens2=tsg_policy_hiddens2,
            encoder=encoder_dict['encoder'],
            with_layer_norm=True,
        )
    )

    qfunc_kwargs = dict(
        obs_action_dim=encoder_hiddens[-1] * 3 + n_extra_cols_append,
        hiddens=qfunc_hiddens, 
        with_layer_norm=True, 
        with_batch_norm=False,
        encoder=None
    )

    Q1_kwargs = qfunc_kwargs.copy()
    Q1_kwargs['encoder'] = encoder_dict['encoderq1']
    Q2_kwargs = qfunc_kwargs.copy()
    Q2_kwargs['encoder'] = encoder_dict['encoderq2']
    Q1t_kwargs = qfunc_kwargs.copy()
    Q1t_kwargs['encoder'] = encoder_dict['encoderq1t']
    Q2t_kwargs = qfunc_kwargs.copy()
    Q2t_kwargs['encoder'] = encoder_dict['encoderq2t']

    agent_kwargs=dict(
        name='SACAgentGraph',
        policy_constructor=policy_constructors[which_policy_kwargs],
        qfunc_constructor=Qfunc,
        env_constructor=GraphEnv,
        buffer_constructor=GraphBuffer,
        optimiser_constructors=dict(
            policy_optim=torch.optim.Adam,
            temperature_optim=torch.optim.Adam,
            Q1_optim=torch.optim.Adam,
            Q2_optim=torch.optim.Adam,
        ),
        entropy_lb=encoder_hiddens[-1],
        policy_lr=1e-3,
        temperature_lr=1e-2,
        qfunc_lr=1e-3,
        tau=0.005,
        discount=1.,
        save_to=TEST_OUTPUTS_PATH,
        cache_best_policy=False,
        clip_grads=False,
        fixed_temperature=None,
        UT_trick=UT_trick,
        with_entropy=False,
    )

    config = dict(
        training_kwargs=dict(
            seed=seed,
            num_iters=100,
            num_steps_to_sample=300,
            num_grad_steps=1,
            batch_size=100,
            num_eval_steps_to_sample=n_nodes,
            min_steps_to_presample=300,
        ),
        Q1_kwargs=Q1_kwargs,
        Q2_kwargs=Q2_kwargs,
        Q1t_kwargs=Q1t_kwargs,
        Q2t_kwargs=Q2t_kwargs,
        policy_kwargs=policy_kwargs[which_policy_kwargs],
        buffer_kwargs=dict(
            max_size=100_000,
            nodes=nodes,
            state_reward=which_reward_fn == 'state_reward_fn',
            seed=seed,
            transform_=transform_,
            drop_repeats_or_self_loops=True,
            graphs_per_batch=100,
            action_is_index=action_is_index,
            action_dim=action_dim,
            per_decision_imp_sample=per_decision_imp_sample,
            reward_scale=reward_scale,
            log_offset=0.,
            lcr_reg=True, 
            verbose=True,
            unnorm_policy=False,
            be_deterministic=False,
        ),
        env_kwargs=dict(
            x=nodes,
            expert_edge_index=None,
            num_edges_start_from=0,
            reward_fn=reward_fn,
            max_episode_steps=n_nodes,
            num_expert_steps=n_nodes,
            max_repeats=n_nodes,
            max_self_loops=n_nodes,
            drop_repeats_or_self_loops=True,
            id=None,
            reward_fn_termination=False,
            calculate_reward=False,
            min_steps_to_do=3,
            similarity_func=sigmoid_similarity,
            forbid_self_loops_repeats=False,
        )
    )

    # get config for the IRL trainer;
    irl_trainer_config = dict(
        num_expert_traj=30,
        graphs_per_batch=config['buffer_kwargs']['graphs_per_batch'],
        num_extra_paths_gen=20,
        num_edges_start_from=config['env_kwargs']['num_edges_start_from'],
        reward_optim_lr_scheduler=None,
        reward_grad_clip=False,
        reward_scale=config['buffer_kwargs']['reward_scale'],
        per_decision_imp_sample=config['buffer_kwargs']['per_decision_imp_sample'],
        weight_scaling_type=weight_scaling_type,
        unnorm_policy=config['buffer_kwargs']['unnorm_policy'],
        add_expert_to_generated=False,
        lcr_regularisation_coef=num_edges_expert,
        mono_regularisation_on_demo_coef=num_edges_expert,
        verbose=True,
        do_dfs_expert_paths=do_dfs_expert_paths,
        num_reward_grad_steps=1,
    )
    return agent_kwargs, config, reward_fn, irl_trainer_config

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"seed is: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # this is to be passed to get_params()
    params_func_config = dict(
        net_hiddens=64,
        encoder_h=[8, 64],
        embed_dim=8,
        bet_on_homophily=False,
        net2_batch_norm=False,
        with_batch_norm=False,
        final_tanh=True,
        action_is_index=True,
        do_dfs_expert_paths=True,
        UT_trick=False,
        per_decision_imp_sample=True,
        weight_scaling_type='abs_max',
    )

    # process extra stuff;
    # NOTE: boolean values are passed as 0 for False and other int for True;
    int_regex = re.compile(r'(^[0-9]+$)')
    if len(sys.argv) > 2:
        for a in sys.argv[2:]:
            n, v = a.split('=')
            if n in params_func_config:
                if re.match(int_regex, v):
                    v = int(v)
                params_func_config[n] = v
                print(n, v)

    # IRL train config;
    get_params_train = partial(get_params, **params_func_config)
    
    # get kwargs;
    agent_kwargs, config, reward_fn, irl_trainer_config = get_params_train()
    
    # init agent for the IRL training;
    agent = SACAgentGraph(
        **agent_kwargs,
        **config
    )
    
    # init IRL trainer;
    irl_trainer = IRLGraphTrainer(
        reward_fn=reward_fn,
        reward_optim=torch.optim.Adam(reward_fn.parameters(), lr=1e-2),
        agent=agent,
        nodes=nodes,
        expert_edge_index=expert_edge_index,
        **irl_trainer_config,
    )

    # get orthogonal init of nets;
    irl_trainer.OI_init_nets()
    
    # extra info to save in pkl after training is done;
    irl_trainer_config['num_iters'] = 30
    irl_trainer_config['policy_epochs'] = 1
    irl_trainer_config['vis_graph'] = True
    irl_trainer_config['log_offset'] = config['buffer_kwargs']['log_offset']
    irl_trainer_config['policy_lr']=agent_kwargs['policy_lr']
    irl_trainer_config['temperature_lr']=agent_kwargs['temperature_lr']
    irl_trainer_config['qfunc_lr']=agent_kwargs['qfunc_lr']
    irl_trainer_config['tau']=agent_kwargs['tau']
    irl_trainer_config['discount']=agent_kwargs['discount']
    irl_trainer_config['fixed_temperature'] = agent_kwargs['fixed_temperature']
    for k, v in params_func_config.items():
        irl_trainer_config[k] = v
    
    # train IRL;
    irl_trainer.train_irl(
        num_iters=irl_trainer_config['num_iters'], 
        policy_epochs=irl_trainer_config['policy_epochs'], 
        vis_graph=irl_trainer_config['vis_graph'], 
        with_pos=False,
        config=irl_trainer_config
    )
    
    # get learned reward;
    reward_fn = irl_trainer.reward_fn
    reward_fn.eval()
    agent_kwargs['save_to'] = None

    # init graph;
    target_edge_index = get_consec_edge_index(barabasi_albert_graph(50, 3))
    target_nodes = trig_circle_init(50)
    target_graph = Data(x=target_nodes, edge_index=target_edge_index)
    params_func_config['n_nodes']= target_nodes.shape[0]
    params_func_config['node_dim']= target_nodes.shape[-1]
    params_func_config['nodes'] = target_nodes
    get_params_eval = partial(get_params, **params_func_config)
    
    # Run experiment suite 1. from GraphOpt paper;
    # this compares graph stats like degrees, triangle, clust coef;
    # from graphs made by 
    # (1) training new policy with learned reward_fn on target graph;
    # (2) deploying learned policy from IRL task directly on source graph;
    save_graph_stats_k_runs_GO1(
        irl_trainer.agent,
        reward_fn, 
        SACAgentGraph, 
        num_epochs_new_policy=15,
        target_graph=target_graph,
        run_k_times=3,
        new_policy_param_getter_fn=get_params_eval,
        sort_metrics=False,
        euc_dist_idxs=None,#torch.tensor([[0, 1]], dtype=torch.long),
        vis_graph=True,
        with_pos=False,
        **agent_kwargs
    )
    
    # currently prints summary stats of graph and saves plot of stats;
    save_analysis_graph_stats(
        irl_trainer.agent.save_to / 'target_graph_stats',
        'graph_stats_hist_kde_plots.png',
        suptitle=None,
        verbose=False
    )
