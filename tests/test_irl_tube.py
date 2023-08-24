import sys
from pathlib import Path
p = Path(__file__).absolute().parent.parent
if sys.path[-1] != str(p):
    sys.path.append(str(p))
print(str(p))

DATA_PATH = p / 'data/Subways'
SOURCE_POSITIONS_PATH = DATA_PATH / 'london-tube/list_stations_positions-clean.txt'
SOURCE_TOPO_PATH = DATA_PATH / 'london-tube/London-topologies/London-2009-adjacency.txt'
# I know there is a typo in the name - liste - but that's how it came;
# I don't want to change naming;
TARGET_POSITIONS_PATH = DATA_PATH / 'Paris/liste_stations_positions.txt'
TARGET_TOPO_PATH = DATA_PATH / 'Paris/Paris-topologies/Paris-2009-adjacency.txt'

from graph_irl.experiments_init_utils import *
from graph_irl.graph_rl_utils import *
from graph_irl.transforms import *
from graph_irl.sac import SACAgentGraph, TEST_OUTPUTS_PATH
from graph_irl.irl_trainer import IRLGraphTrainer
from graph_irl.eval_metrics import *
from urban_nets_utils import *

from functools import partial
import re

from torch_geometric.data import Data

import random
import numpy as np
import torch


# graph transform setup;
transform_fn_ = None
n_cols_append = 0  # based on transform_fn_

get_graph_level_feats_fn = None
n_extra_cols_append = 0  # based on get_graph_level_feats_fn

# instantiate transform;
transform_ = None

# this is to be passed to get_params()
params_func_config = dict(
    num_iters=100,
    batch_size=100,
    graphs_per_batch=100,
    num_grad_steps=1,
    reward_scale=1.,
    net_hiddens=[64],
    encoder_hiddens=[64],
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
    n_cols_append=n_cols_append,
    n_extra_cols_append=n_extra_cols_append,
    ortho_init=True,
    seed=0,
    transform_=transform_,
    clip_grads=False,
    fixed_temperature=None,
    num_steps_to_sample=None,
    unnorm_policy=False,
)


if __name__ == "__main__":
    # set training params from stdin;
    # NOTE: booleans are passed as 0 for False and other int for True;
    params_func_config = arg_parser(params_func_config, sys.argv)
    
    # set the seed;
    random.seed(params_func_config['seed'])
    np.random.seed(params_func_config['seed'])
    torch.manual_seed(params_func_config['seed'])

    # init graphs now, not to mess up the seed if 
    # experiments are with different NN architectures
    # since diff layer sizes require different num of 
    # random param inits;

    scaling = get_min_max_scaled_feats
    graph_source = get_city_graph(SOURCE_POSITIONS_PATH,
                                  SOURCE_TOPO_PATH,
                                  scaling)
    
    # target graph;
    graph_target = get_city_graph(TARGET_POSITIONS_PATH,
                                 TARGET_TOPO_PATH,
                                 scaling)

    # add the graph parameters;
    params_func_config['n_nodes'] = graph_source.x.shape[0]
    params_func_config['node_dim'] = graph_source.x.shape[-1]
    params_func_config['nodes'] = graph_source.x
    params_func_config['num_edges_expert'] = graph_source.edge_index.shape[-1] // 2

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
        nodes=graph_source.x,
        expert_edge_index=graph_source.edge_index,
        **irl_trainer_config,
    )

    # save metrics of source graph;
    names_of_stats = [
        'triangles',
        'degrees',
        'clustcoefs'
    ]
    target_graph_dir = agent.save_to / 'target_graph_stats'
    save_stats_edge_index(
        agent.save_to, names_of_stats, graph_source, 'sourcegraph_'
    )
    
    # save edge_index of target graph, since its stats 
    # are saved in the eval loop;
    with open(agent.save_to / 'targetgraph_edgeidx.pkl', 'wb') as f:
        pickle.dump(graph_target.edge_index.tolist(), f)
    
    # extra info to save in pkl after training is done;
    irl_trainer_config['irl_iters'] = 12
    irl_trainer_config['policy_epochs'] = 1
    irl_trainer_config['vis_graph'] = False
    irl_trainer_config['save_edge_index'] = True
    irl_trainer_config['log_offset'] = config['buffer_kwargs']['log_offset']
    irl_trainer_config['policy_lr']=agent_kwargs['policy_lr']
    irl_trainer_config['temperature_lr']=agent_kwargs['temperature_lr']
    irl_trainer_config['qfunc_lr']=agent_kwargs['qfunc_lr']
    irl_trainer_config['tau']=agent_kwargs['tau']
    irl_trainer_config['discount']=agent_kwargs['discount']
    irl_trainer_config['fixed_temperature'] = agent_kwargs['fixed_temperature']
    for k, v in params_func_config.items():
        irl_trainer_config[k] = v
    
    # start IRL training;
    print(f"IRL training for {len(graph_source.x)}-node graph")
    
    # train IRL;
    irl_trainer.train_irl(
        num_iters=irl_trainer_config['irl_iters'], 
        policy_epochs=irl_trainer_config['policy_epochs'], 
        vis_graph=irl_trainer_config['vis_graph'], 
        save_edge_index=irl_trainer_config['save_edge_index'],
        with_pos=False,
        config=irl_trainer_config
    )
    
    # get learned reward;
    reward_fn = irl_trainer.reward_fn
    reward_fn.eval()
    agent_kwargs['save_to'] = None

    # set relevant target graphs params;
    params_func_config['n_nodes']= graph_target.x.shape[0]
    params_func_config['node_dim']= graph_target.x.shape[-1]
    params_func_config['nodes'] = graph_target.x
    params_func_config['num_edges_expert'] = graph_target.edge_index.shape[-1] // 2
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
        num_epochs_new_policy=5,
        target_graph=graph_target,
        run_k_times=3,
        new_policy_param_getter_fn=get_params_eval,
        sort_metrics=False,
        euc_dist_idxs=None,#torch.tensor([[0, 1]], dtype=torch.long),
        save_edge_index=True,
        vis_graph=False,
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
