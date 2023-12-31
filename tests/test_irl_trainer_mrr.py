import sys
from pathlib import Path
p = Path(__file__).absolute().parent.parent
if sys.path[-1] != str(p):
    sys.path.append(str(p))
print(str(p))

from graph_irl.experiments_init_utils import *
from graph_irl.graph_rl_utils import *
from graph_irl.transforms import *
from graph_irl.sac import SACAgentGraph, TEST_OUTPUTS_PATH
from graph_irl.sac_GO import SACAgentGO
from graph_irl.examples.circle_graph import create_circle_graph
from graph_irl.irl_trainer import IRLGraphTrainer
from graph_irl.eval_metrics import *

from functools import partial

from torch_geometric.data import Data

import random
import numpy as np
import torch


# circular graph;
n_nodes, node_dim = 15, 2
num_edges_expert = n_nodes

# graph transform setup;
transform_fn_ = partial(append_distances_, with_degrees=True)
n_cols_append = 2  # based on transform_fn_

# get_graph_level_feats_fn = partial(
#     get_columns_aggr, col_idxs=[-1], aggr='max', 
#     check_in_dim=node_dim + n_cols_append
# )
get_graph_level_feats_fn = None
n_extra_cols_append = 0#1  # based on get_graph_level_feats_fn

col_names_ridxs = {
    'degrees': -2,
    'euclid2d_distance': -1
}

# instantiate transform;
transform_ = InplaceBatchNodeFeatTransform(
    node_dim, transform_fn_, n_cols_append, 
    col_names_ridxs, n_extra_cols_append, get_graph_level_feats_fn
)

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

    # init graph;
    nodes, expert_edge_index = create_circle_graph(n_nodes, node_dim, trig_circle_init)
    graph_source = Data(x=nodes, edge_index=expert_edge_index)

    # Subsample the edge index and train IRL on it;
    # the remaining subset is used for link prediction eval;
    train_edge_index, positives_dict = split_edge_index(expert_edge_index, .1)
    print("shape of train index and len of positives dict",
            train_edge_index.shape, len(positives_dict))
    
    # add the graph parameters;
    params_func_config['n_nodes'] = n_nodes
    params_func_config['node_dim'] = node_dim
    params_func_config['nodes'] = nodes
    params_func_config['num_edges_expert'] = train_edge_index.shape[-1] // 2
    params_func_config['expert_edge_index'] = train_edge_index
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
        nodes=nodes,
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
    
    irl_trainer_config['multitask_gnn'] = agent_kwargs['with_multitask_gnn_loss']
    irl_trainer_config['irl_iters'] = 2
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
    print(f"IRL training for {n_nodes}-node graph")

    # train IRL;
    irl_trainer.train_irl(
        num_iters=irl_trainer_config['irl_iters'], 
        policy_epochs=irl_trainer_config['policy_epochs'], 
        vis_graph=irl_trainer_config['vis_graph'], 
        save_edge_index=irl_trainer_config['save_edge_index'],
        with_pos=True,  # check the map;
        config=irl_trainer_config
    )

    save_mrr_and_avg(
        agent, train_edge_index, positives_dict, graph_source,
        k_proposals=25,
        do_graphopt=params_func_config['do_graphopt']
    )
