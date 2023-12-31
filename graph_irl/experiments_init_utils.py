import numpy as np
import torch
from typing import Optional
import re

from graph_irl.policy import *
from graph_irl.graph_rl_utils import *
from graph_irl.reward import *
from graph_irl.buffer_v2 import GraphBuffer

from torch_geometric.utils import barabasi_albert_graph, degree
from torch_geometric.nn import knn_graph
import warnings


def split_edge_index(edge_index, test_prop):
    T = edge_index.shape[-1] // 2
    idxs = np.random.choice(
        range(0, edge_index.shape[-1], 2),
        size=max(int(T * test_prop), 3),
        replace=False,
    )
    test_idxs = set(sum([(i, i + 1) for i in idxs], ()))
    msk = np.array([i in test_idxs for i in range(edge_index.shape[-1])])
    train_edge_index = edge_index[:, ~msk]
    positives_dict = {
        (min(edge_index[:, i]).item(), max(edge_index[:, i]).item()): 0
        for i in idxs
    }
    return train_edge_index, positives_dict


def trig_circle_init(*args):
    n_nodes = args[0]
    inputs = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    inputs = torch.tensor(inputs, dtype=torch.float32).view(-1, 1)
    return torch.cat((torch.cos(inputs), torch.sin(inputs)), -1)


def get_consec_edge_index(edge_index):
    edge_set = set()
    new_index = torch.zeros(edge_index.shape, dtype=torch.long)
    j = 0
    for i in range(edge_index.shape[-1]):
        first, second = (
            edge_index[:, i].min().item(),
            edge_index[:, i].max().item(),
        )
        if (first, second) in edge_set:
            continue
        edge_set.add((first, second))
        new_index[0, j] = first
        new_index[1, j] = second
        new_index[0, j + 1] = second
        new_index[1, j + 1] = first
        j += 2
    return new_index


params_func_config = dict(
    num_iters=100,
    batch_size=100,
    graphs_per_batch=100,
    num_grad_steps=1,
    num_steps_to_sample=None,
    reward_scale=1.0,
    net_hiddens=[64],
    encoder_hiddens=[64],
    embed_dim=8,
    final_tanh=True,
    action_is_index=True,
    do_dfs_expert_paths=False,
    UT_trick=False,
    per_decision_imp_sample=True,
    weight_scaling_type="abs_max",
    n_cols_append=None,
    n_extra_cols_append=None,
    ortho_init=False,
    seed=0,
    transform_=None,
    clip_grads=False,
    fixed_temperature=None,
    unnorm_policy=False,
    with_multitask_gnn_loss=False,
    multitask_coef=1.0,
    max_size=10_000,
    with_mlp_batch_norm=True,
    with_mlp_layer_norm=False,
    heads=1,
    do_graphopt=False,
    no_q_encoder=False,
    policy_lr=1e-3,
    temperature_lr=1e-3,
    qfunc_lr=1e-3,
    reward_lr=1e-2,
    log_sigma_min=None,  # -20,
    log_sigma_max=None,  # 2,
    use_valid_samples=False,
    with_knn_msgs=False,
    with_lcr=True,
    with_mono=True,
    zero_interm_rew=False,
    quad_reward_penalty_coef=None,
    reward_l2_coef=None,
)


def get_params(
    n_nodes,
    node_dim,
    nodes,
    num_edges_expert,
    expert_edge_index,
    num_iters=100,
    batch_size=100,
    graphs_per_batch=100,
    num_grad_steps=1,
    num_steps_to_sample=None,
    reward_scale=1.0,
    net_hiddens: list = [64],
    encoder_hiddens: list = [64],
    embed_dim: int = 8,
    final_tanh=True,
    action_is_index=True,
    do_dfs_expert_paths=False,
    UT_trick=False,
    per_decision_imp_sample=True,
    weight_scaling_type="abs_max",
    n_cols_append=None,
    n_extra_cols_append=None,
    ortho_init=False,
    seed=0,
    transform_=None,
    clip_grads=False,
    fixed_temperature: Optional[int] = None,
    unnorm_policy=False,
    with_multitask_gnn_loss=False,
    multitask_coef=1.0,
    max_size=10_000,
    with_mlp_batch_norm=True,
    with_mlp_layer_norm=False,
    heads=1,
    do_graphopt=False,
    no_q_encoder=False,
    policy_lr=1e-3,
    temperature_lr=1e-3,
    qfunc_lr=1e-3,
    reward_lr=1e-2,
    log_sigma_min=None,  # -20,
    log_sigma_max=None,  # 2,
    use_valid_samples=False,
    with_knn_msgs=False,
    with_lcr=True,
    with_mono=True,
    zero_interm_rew=False,
    quad_reward_penalty_coef=None,
    reward_l2_coef=None,
):
    knn_edge_index = None
    if with_knn_msgs:
        k = (
            degree(expert_edge_index[0], num_nodes=n_nodes)
            .mean()
            .int()
            .item()
        )
        knn_edge_index = knn_graph(nodes, k=k)
    # if we do multitask loss for gnn, make sure nothing gets
    # appended to the graph level embedding for now;
    if with_multitask_gnn_loss:
        assert n_extra_cols_append == 0 or n_extra_cols_append is None
        assert n_cols_append > 0
    print(n_nodes, node_dim, nodes.shape, num_edges_expert)
    # some setup;
    if num_steps_to_sample is None:
        num_steps_to_sample = max(num_edges_expert, 100)
    if isinstance(encoder_hiddens, int):
        encoder_hiddens = [encoder_hiddens]
    if isinstance(net_hiddens, int):
        net_hiddens = [net_hiddens]
    encoder_hiddens = encoder_hiddens + [embed_dim]
    reward_fn_hiddens = net_hiddens
    gauss_policy_hiddens = net_hiddens
    tsg_policy_hiddens1 = net_hiddens
    tsg_policy_hiddens2 = net_hiddens
    qfunc_hiddens = net_hiddens
    which_reward_fn = "state_reward_fn"
    which_policy_kwargs = "tanh_gauss_policy_kwargs"
    action_dim = embed_dim * 2

    encoder_dict = dict(
        encoder=GCN(
            node_dim + n_cols_append,
            encoder_hiddens,
            heads=heads,
            final_tanh=final_tanh,
            knn_edge_index=knn_edge_index,
        ),
        encoderq1=GCN(
            node_dim + n_cols_append,
            encoder_hiddens,
            heads=heads,
            final_tanh=final_tanh,
            knn_edge_index=knn_edge_index,
        ),
        encoderq2=GCN(
            node_dim + n_cols_append,
            encoder_hiddens,
            heads=heads,
            final_tanh=final_tanh,
            knn_edge_index=knn_edge_index,
        ),
        encoderq1t=GCN(
            node_dim + n_cols_append,
            encoder_hiddens,
            heads=heads,
            final_tanh=final_tanh,
            knn_edge_index=knn_edge_index,
        ),
        encoderq2t=GCN(
            node_dim + n_cols_append,
            encoder_hiddens,
            heads=heads,
            final_tanh=final_tanh,
            knn_edge_index=knn_edge_index,
        ),
        encoder_reward=GCN(
            node_dim + n_cols_append,
            encoder_hiddens,
            heads=heads,
            final_tanh=final_tanh,
            knn_edge_index=knn_edge_index,
        ),
    )

    reward_funcs = dict(
        reward_fn=GraphReward(
            encoder_dict["encoder_reward"],
            embed_dim=embed_dim + n_extra_cols_append,
            hiddens=reward_fn_hiddens,
            with_batch_norm=with_mlp_batch_norm,
            with_layer_norm=with_mlp_layer_norm,
        ),
        state_reward_fn=StateGraphReward(
            None if do_graphopt else encoder_dict["encoder_reward"],
            embed_dim=embed_dim + n_extra_cols_append,
            hiddens=reward_fn_hiddens,
            with_batch_norm=with_mlp_batch_norm,
            with_layer_norm=with_mlp_layer_norm,
        ),
    )

    reward_fn = reward_funcs[which_reward_fn]

    policy_constructors = dict(
        tsg_policy_kwargs=TwoStageGaussPolicy,
        gauss_policy_kwargs=GaussPolicy,
        tanh_gauss_policy_kwargs=TanhGaussPolicy,
    )

    policy_kwargs = dict(
        gauss_policy_kwargs=dict(
            obs_dim=embed_dim + n_extra_cols_append,
            action_dim=embed_dim,
            hiddens=gauss_policy_hiddens,
            with_batch_norm=with_mlp_batch_norm,
            with_layer_norm=with_mlp_layer_norm,
            encoder=encoder_dict["encoder"],
            two_action_vectors=True,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
        ),
        tanh_gauss_policy_kwargs=dict(
            obs_dim=embed_dim + n_extra_cols_append,
            action_dim=embed_dim,
            hiddens=gauss_policy_hiddens,
            with_batch_norm=with_mlp_batch_norm,
            with_layer_norm=with_mlp_layer_norm,
            encoder=encoder_dict["encoder"],
            two_action_vectors=True,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
        ),
        tsg_policy_kwargs=dict(
            obs_dim=embed_dim + n_extra_cols_append,
            action_dim=embed_dim,
            hiddens1=tsg_policy_hiddens1,
            hiddens2=tsg_policy_hiddens2,
            encoder=encoder_dict["encoder"],
            with_batch_norm=with_mlp_batch_norm,
            with_layer_norm=with_mlp_layer_norm,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
        ),
    )
    if UT_trick and with_mlp_batch_norm:
        warnings.warn(
            "Qfunc cannot implement UT trick with batch norm on. "
            "This is because of the extra dimension created for "
            "the sigma points. BN will start treating this as the "
            "channel dimension which may be different (and usually is) "
            "from the intended dim of the BN layers in the MLP."
        )
    qfunc_kwargs = dict(
        obs_action_dim=embed_dim * 3 + n_extra_cols_append,
        hiddens=qfunc_hiddens,
        with_layer_norm=with_mlp_layer_norm,
        # cant do UT trick with batch norm since batch_norm will
        # need to be applied to 2d + 1 nums, but it is init to handle
        # the output of the previous affine layer - so an error for
        # the dim will be raised;
        with_batch_norm=False if UT_trick else with_mlp_batch_norm,
        encoder=None,
    )

    Q1_kwargs = qfunc_kwargs.copy()
    Q2_kwargs = qfunc_kwargs.copy()
    Q1t_kwargs = qfunc_kwargs.copy()
    Q2t_kwargs = qfunc_kwargs.copy()
    if not no_q_encoder:
        assert not do_graphopt
        Q1_kwargs["encoder"] = encoder_dict["encoderq1"]
        Q2_kwargs["encoder"] = encoder_dict["encoderq2"]
        Q1t_kwargs["encoder"] = encoder_dict["encoderq1t"]
        Q2t_kwargs["encoder"] = encoder_dict["encoderq2t"]

    agent_name = "SACAgentGO" if do_graphopt else "SACAgentGraph"
    agent_name = agent_name + (
        f"-nh-{len(net_hiddens)}x{net_hiddens[0]}"
        f"-eh-{len(encoder_hiddens)}x{encoder_hiddens[0]}"
        f"-embdim-{embed_dim}"
    )
    agent_kwargs = dict(
        name=agent_name,
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
        entropy_lb=embed_dim,
        policy_lr=policy_lr,
        temperature_lr=temperature_lr,
        qfunc_lr=qfunc_lr,
        tau=0.005,
        discount=1.0,
        save_to=TEST_OUTPUTS_PATH,
        cache_best_policy=False,
        clip_grads=clip_grads,
        fixed_temperature=fixed_temperature,
        UT_trick=UT_trick,
        with_entropy=False,
        with_multitask_gnn_loss=with_multitask_gnn_loss,
        multitask_coef=multitask_coef,
        no_q_encoder=no_q_encoder,
        use_valid_samples=use_valid_samples,
        zero_interm_rew=zero_interm_rew,
    )

    config = dict(
        training_kwargs=dict(
            seed=seed,
            num_iters=num_iters,
            num_steps_to_sample=num_steps_to_sample,
            num_grad_steps=num_grad_steps,
            batch_size=batch_size,
            num_eval_steps_to_sample=num_edges_expert,
            min_steps_to_presample=max(300, num_edges_expert),
        ),
        Q1_kwargs=Q1_kwargs,
        Q2_kwargs=Q2_kwargs,
        Q1t_kwargs=Q1t_kwargs,
        Q2t_kwargs=Q2t_kwargs,
        policy_kwargs=policy_kwargs[which_policy_kwargs],
        buffer_kwargs=dict(
            max_size=max_size,
            nodes=nodes,
            state_reward=which_reward_fn == "state_reward_fn",
            seed=seed,
            transform_=transform_,
            drop_repeats_or_self_loops=True,
            graphs_per_batch=graphs_per_batch,
            action_is_index=action_is_index,
            action_dim=action_dim,
            per_decision_imp_sample=per_decision_imp_sample,
            reward_scale=reward_scale,
            log_offset=0.0,
            lcr_reg=with_lcr,
            verbose=True,
            unnorm_policy=unnorm_policy,
            be_deterministic=False,
            zero_interm_rew=zero_interm_rew,
        ),
        env_kwargs=dict(
            x=nodes,
            expert_edge_index=None,
            num_edges_start_from=0,
            reward_fn=reward_fn,
            max_episode_steps=num_edges_expert,
            num_expert_steps=num_edges_expert,
            max_repeats=num_edges_expert,  # as much as longest possible path;
            max_self_loops=num_edges_expert,
            drop_repeats_or_self_loops=True,
            id=None,
            reward_fn_termination=False,
            calculate_reward=False,
            min_steps_to_do=3,
            similarity_func=sigmoid_similarity,
        ),
    )

    # get config for the IRL trainer;
    irl_trainer_config = dict(
        num_expert_traj=30,
        graphs_per_batch=graphs_per_batch,
        num_extra_paths_gen=20,
        num_edges_start_from=config["env_kwargs"]["num_edges_start_from"],
        reward_optim_lr_scheduler=None,
        reward_grad_clip=clip_grads,
        reward_scale=reward_scale,
        per_decision_imp_sample=config["buffer_kwargs"][
            "per_decision_imp_sample"
        ],
        weight_scaling_type=weight_scaling_type,
        unnorm_policy=config["buffer_kwargs"]["unnorm_policy"],
        add_expert_to_generated=False,
        lcr_regularisation_coef=num_edges_expert
        - config["env_kwargs"]["num_edges_start_from"]
        if with_lcr
        else None,
        mono_regularisation_on_demo_coef=num_edges_expert
        - config["env_kwargs"]["num_edges_start_from"]
        if with_mono
        else None,
        verbose=True,
        do_dfs_expert_paths=do_dfs_expert_paths,
        num_reward_grad_steps=1,
        ortho_init=ortho_init,
        do_graphopt=do_graphopt,
        zero_interm_rew=zero_interm_rew,
        quad_reward_penalty=quad_reward_penalty_coef,
        reward_l2_coef=reward_l2_coef,
    )
    return agent_kwargs, config, reward_fn, irl_trainer_config


def arg_parser(settable_params, argv):
    print(
        "\n------------------: USAGE :------------------\n"
        f"\nPass kwargs as name=value.\n"
        "If name is valid, value will be set.\n"
        "Valid kwargs to the script are:\n",
        settable_params.keys(),
        end="\n\n",
    )
    int_list_regex = re.compile("^([0-9]+,)+[0-9]+$")
    int_regex = re.compile("^(-?)[0-9]+[0-9]*$")
    if len(argv) > 1:
        for a in argv[1:]:
            n, v = a.split("=")
            if n in settable_params:
                if re.match(int_list_regex, v):
                    nums = v.split(",")
                    v = [int(temp) for temp in nums]
                elif re.match(int_regex, v):
                    v = int(v)
                elif "_coef" in n or "_lr" in n:
                    v = float(v)
                settable_params[n] = v
                print(f"{n}={v}", type(v), v)
            else:
                print(f"{n} not a valid argument")
    print("\n")
    return settable_params


def unif_init(n_nodes, n_dim=2):
    return torch.distributions.Uniform(0.0, 1.0).sample((n_nodes, n_dim))


def get_ba_graph(num_nodes, num_edges, node_feat_init_fn=None):
    if node_feat_init_fn is None:
        node_feat_init_fn = trig_circle_init
    nodes = node_feat_init_fn(num_nodes)
    edges = barabasi_albert_graph(num_nodes, num_edges)
    return Data(x=nodes, edge_index=get_consec_edge_index(edges))
