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

    scaling = get_min_max_scaled_feats
    graph_source = get_city_graph(SOURCE_POSITIONS_PATH,
                                  SOURCE_TOPO_PATH,
                                  scaling)
    
    # Subsample the edge index and train IRL on it;
    # the remaining subset is used for link prediction eval;
    train_edge_index, positives_dict = split_edge_index(
        graph_source.edge_index, .1
    )
    print("shape of train index and len of positives dict",
            train_edge_index.shape, len(positives_dict))

    # add the graph parameters;
    params_func_config['n_nodes'] = graph_source.x.shape[0]
    params_func_config['node_dim'] = graph_source.x.shape[-1]
    params_func_config['nodes'] = graph_source.x
    params_func_config['num_edges_expert'] = train_edge_index.shape[-1] // 2

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
    irl_trainer_config['irl_iters'] = 16
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
        if k == 'nodes':
            continue
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
    
    # make env start from subsampled edge index and sample 
    # remaining num of edges;
    agent.env.expert_edge_index = train_edge_index
    agent.env.num_edges_start_from = train_edge_index.shape[-1] // 2
    num_expert_steps = graph_source.edge_index.shape[-1] // 2
    agent.env.spec.max_episode_steps = num_expert_steps
    agent.env.num_expert_steps = num_expert_steps
    agent.env.max_repeats = agent.env.max_self_loops = num_expert_steps

    mrr_stats, found_rate_stats = get_mrr_and_avg(
        agent, agent.env, positives_dict
    )
    mrr_stats_dir = agent.save_to / 'mrr_stats'
    if not mrr_stats_dir.exists():
        mrr_stats_dir.mkdir()
    with open(mrr_stats_dir / 'mrr_stats.pkl', 'wb') as f:
        pickle.dump(mrr_stats, f)
    with open(mrr_stats_dir / 'found_rates.pkl', 'wb') as f:
        pickle.dump(found_rate_stats, f)
    print("MRR stats:", mrr_stats, 
          "found rates stats:", found_rate_stats, sep='\n')
    