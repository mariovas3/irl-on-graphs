import sys
from pathlib import Path
p = Path(__file__).absolute().parent.parent
if sys.path[-1] != str(p):
    sys.path.append(str(p))
print(str(p))

from graph_irl.graph_rl_utils import *
from graph_irl.transforms import *
from graph_irl.sac import SACAgentGraph, TEST_OUTPUTS_PATH
from graph_irl.sac_GO import SACAgentGO
from graph_irl.irl_trainer import IRLGraphTrainer
from graph_irl.eval_metrics import *
from graph_irl.experiments_init_utils import *

from functools import partial

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
params_func_config['n_cols_append'] = n_cols_append
params_func_config['n_extra_cols_append'] = n_extra_cols_append
params_func_config['transform_'] = transform_


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

    # source graph;
    n_nodes_source = 40
    n_edges_source = 3
    graph_source = get_ba_graph(n_nodes_source, n_edges_source)

    # Subsample the edge index and train IRL on it;
    # the remaining subset is used for link prediction eval;
    train_edge_index, positives_dict = split_edge_index(
        graph_source.edge_index, .1
    )
    print("shape of train index and len of positives dict",
            train_edge_index.shape, len(positives_dict))
    
    # add the graph parameters;
    params_func_config['n_nodes'] = n_nodes_source
    params_func_config['node_dim'] = graph_source.x.shape[-1]
    params_func_config['nodes'] = graph_source.x
    params_func_config['num_edges_expert'] = train_edge_index.shape[-1] // 2
    params_func_config['expert_edge_index'] = train_edge_index

    # IRL train config;
    get_params_train = partial(get_params, **params_func_config)

    if params_func_config['do_graphopt']:
        agent_constructor = SACAgentGO
    else:
        agent_constructor = SACAgentGraph
    
    # get kwargs;
    agent_kwargs, config, reward_fn, irl_trainer_config = get_params_train()
    
    # init agent for the IRL training;
    agent = agent_constructor(
        **agent_kwargs,
        **config
    )
    
    # init IRL trainer;
    irl_trainer = IRLGraphTrainer(
        reward_fn=reward_fn,
        reward_optim=torch.optim.Adam(
            reward_fn.parameters(), 
            lr=params_func_config['reward_lr']
        ),
        agent=agent,
        nodes=graph_source.x,
        expert_edge_index=train_edge_index,
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
    
    # extra info to save in pkl after training is done;
    irl_trainer_config['multitask_gnn'] = agent_kwargs['with_multitask_gnn_loss']
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
        if k in ('nodes', 'transform_', 'expert_edge_index'):
            continue
        irl_trainer_config[k] = v
    
    # start IRL training;
    print(f"IRL training for {n_nodes_source}-node graph")
    
    # train IRL;
    irl_trainer.train_irl(
        num_iters=irl_trainer_config['irl_iters'], 
        policy_epochs=irl_trainer_config['policy_epochs'], 
        vis_graph=irl_trainer_config['vis_graph'], 
        save_edge_index=irl_trainer_config['save_edge_index'],
        with_pos=False,
        config=irl_trainer_config
    )

    save_mrr_and_avg(
        agent, train_edge_index, positives_dict, graph_source,
        k_proposals=25,
        do_graphopt=params_func_config['do_graphopt'],
    )
