import sys
from pathlib import Path
p = Path(__file__).absolute().parent.parent
if sys.path[-1] != str(p):
    sys.path.append(str(p))
print(str(p))

from graph_irl.buffer_v2 import GraphBuffer
from graph_irl.policy import GaussPolicy, TwoStageGaussPolicy, GCN, Qfunc
from graph_irl.graph_rl_utils import GraphEnv
from graph_irl.sac import SACAgentGraph, TEST_OUTPUTS_PATH
from graph_irl.reward import GraphReward, StateGraphReward
from graph_irl.examples.circle_graph import create_circle_graph
from graph_irl.irl_trainer import IRLGraphTrainer

import random
import numpy as np
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def uniform_init(size_t):
    return torch.distributions.Uniform(0., 1.).sample(size_t)


# circular graph with 7 nodes;
n_nodes, node_dim = 11, 8
nodes, expert_edge_index = create_circle_graph(n_nodes, node_dim, torch.randn)
# nodes = torch.ones_like(nodes)  # doesn't seem to work for ones;
print(nodes, expert_edge_index)
encoder_hiddens = [256, 256, 8]
reward_fn_hiddens = [256, 256]
gauss_policy_hiddens = [256, 256]
tsg_policy_hiddens1 = [256, 256]
tsg_policy_hiddens2 = [256]
qfunc_hiddens = [256, 256]
which_reward_fn = 'state_reward_fn'
which_policy_kwargs = 'gauss_policy_kwargs'
action_is_index = False
action_dim = encoder_hiddens[-1] * 2
reward_scale = encoder_hiddens[-1]

print(f"IRL training for {n_nodes}-node graph")

def get_params():
    encoder_dict = dict(
        encoder = GCN(node_dim, encoder_hiddens, with_layer_norm=True, final_tanh=True),
        encoderq1 = GCN(node_dim, encoder_hiddens, with_layer_norm=True, final_tanh=True),
        encoderq2 = GCN(node_dim, encoder_hiddens, with_layer_norm=True, final_tanh=True),
        encoderq1t = GCN(node_dim, encoder_hiddens, with_layer_norm=True, final_tanh=True),
        encoderq2t = GCN(node_dim, encoder_hiddens, with_layer_norm=True, final_tanh=True),
        encoder_reward = GCN(node_dim, encoder_hiddens, with_layer_norm=True, final_tanh=True),
    )


    reward_funcs = dict(
        reward_fn = GraphReward(
            encoder_dict['encoder_reward'], 
            encoder_hiddens[-1], 
            hiddens=reward_fn_hiddens, 
            with_layer_norm=True,
            with_batch_norm=False,
        ),
        state_reward_fn=StateGraphReward(
            encoder_dict['encoder_reward'], 
            encoder_hiddens[-1], 
            hiddens=reward_fn_hiddens, 
            with_layer_norm=True,
            with_batch_norm=False,
        )
    )

    reward_fn = reward_funcs[which_reward_fn]

    policy_constructors = dict(
        tsg_policy_kwargs=TwoStageGaussPolicy,
        gauss_policy_kwargs=GaussPolicy
    )
    
    policy_kwargs = dict(
        gauss_policy_kwargs = dict(
            obs_dim=encoder_hiddens[-1],
            action_dim=encoder_hiddens[-1],
            hiddens=gauss_policy_hiddens,
            with_layer_norm=True,
            encoder=encoder_dict['encoder'],
            two_action_vectors=True,
        ),
        tsg_policy_kwargs = dict(
            obs_dim=encoder_hiddens[-1],
            action_dim=encoder_hiddens[-1],
            hiddens1=tsg_policy_hiddens1,
            hiddens2=tsg_policy_hiddens2,
            encoder=encoder_dict['encoder'],
            with_layer_norm=True,
        )
    )

    qfunc_kwargs = dict(
        obs_action_dim=encoder_hiddens[-1] * 3, 
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
        policy_lr=3e-4,
        temperature_lr=3e-4,
        qfunc_lr=1e-3, #3e-4,
        tau=0.005,
        discount=1.,
        save_to=TEST_OUTPUTS_PATH,
        cache_best_policy=False,
        clip_grads=False,
        zero_temperature=False,
        UT_trick=True,
        with_entropy=True,
    )

    config = dict(
        training_kwargs=dict(
            seed=0,
            num_iters=50,
            num_steps_to_sample=100,
            num_grad_steps=1,
            batch_size=100,
            num_eval_steps_to_sample=n_nodes,
            min_steps_to_presample=0,
        ),
        Q1_kwargs=Q1_kwargs,
        Q2_kwargs=Q2_kwargs,
        Q1t_kwargs=Q1t_kwargs,
        Q2t_kwargs=Q2t_kwargs,
        policy_kwargs=policy_kwargs[which_policy_kwargs],
        buffer_kwargs=dict(
            max_size=10_000,
            nodes=nodes,
            state_reward=which_reward_fn == 'state_reward_fn',
            seed=0,
            drop_repeats_or_self_loops=True,
            graphs_per_batch=100,
            action_is_index=action_is_index,
            action_dim=action_dim,
            per_decision_imp_sample=True,
            reward_scale=reward_scale,
            log_offset=0.,
            lcr_reg=False, 
            verbose=True,
            unnorm_policy=True,
            be_deterministic=False,
        ),
        env_kwargs=dict(
            x=nodes,
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
        )
    )
    return agent_kwargs, config, reward_fn

agent_kwargs, config, reward_fn = get_params()

agent = SACAgentGraph(
    **agent_kwargs,
    **config
)

irl_trainer_config = dict(
    num_expert_traj=10,
    graphs_per_batch=config['buffer_kwargs']['graphs_per_batch'],
    num_extra_paths_gen=5,
    reward_optim_lr_scheduler=None,
    reward_grad_clip=False,
    reward_scale=config['buffer_kwargs']['reward_scale'],
    per_decision_imp_sample=config['buffer_kwargs']['per_decision_imp_sample'],
    unnorm_policy=config['buffer_kwargs']['unnorm_policy'],
    add_expert_to_generated=False,
    lcr_regularisation_coef=None,
    mono_regularisation_on_demo_coef=1 / (expert_edge_index.shape[-1] // 2),
    verbose=True,
)

irl_trainer = IRLGraphTrainer(
    reward_fn=reward_fn,
    reward_optim=torch.optim.Adam(reward_fn.parameters(), lr=1e-2),
    agent=agent,
    nodes=nodes,
    expert_edge_index=expert_edge_index,
    **irl_trainer_config,
)

irl_trainer_config['num_iters'] = 12
irl_trainer_config['policy_epochs'] = 1
irl_trainer_config['vis_graph'] = True
irl_trainer_config['log_offset'] = config['buffer_kwargs']['log_offset']
irl_trainer_config['policy_lr']=agent_kwargs['policy_lr']
irl_trainer_config['temperature_lr']=agent_kwargs['temperature_lr']
irl_trainer_config['qfunc_lr']=agent_kwargs['qfunc_lr']
irl_trainer_config['tau']=agent_kwargs['tau']
irl_trainer_config['discount']=agent_kwargs['discount']
irl_trainer_config['zero_temperature'] = agent_kwargs['zero_temperature']


irl_trainer.train_irl(
    num_iters=irl_trainer_config['num_iters'], 
    policy_epochs=irl_trainer_config['policy_epochs'], 
    vis_graph=irl_trainer_config['vis_graph'], 
    config=irl_trainer_config
)


agent_kwargs, config, reward_fn = get_params()
reward_fn = irl_trainer.reward_fn
reward_fn.eval()
config['env_kwargs']['reward_fn'] = reward_fn

agent = SACAgentGraph(
    **agent_kwargs,
    **config
)

agent.buffer.verbose = False
agent.tau = 0.005

agent.train_k_epochs(k=10, vis_graph=True)

