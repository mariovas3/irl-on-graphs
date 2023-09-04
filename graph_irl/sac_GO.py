import numpy as np
import torch
from torch import nn
from graph_irl.sac import track_params, SACAgentGraph
from graph_irl.graph_rl_utils import get_action_vector_from_idx, get_valid_proposal
from graph_irl.distributions import batch_UT_trick_from_samples
import torch_geometric.nn as tgnn

from copy import deepcopy
from tqdm import tqdm
import math


class SACAgentGO(SACAgentGraph):
    def __init__(self, **kwargs):
        super(SACAgentGO, self).__init__(**kwargs)
        
        # this will be used to calculate rewards at batch-sampling
        # time and will not change during SAC training. It will 
        # instead be updated after each SAC train_k_epochs call;
        self.old_encoder = deepcopy(self.policy.encoder)
        self.old_encoder.requires_grad_(False)
    
    def train_k_epochs(
        self, k, *args, config=None, vis_graph=False, 
        with_pos=False, save_edge_index=False, **kwargs
    ):
        super().train_k_epochs(
            k, *args, config=config, vis_graph=vis_graph, 
            with_pos=with_pos, save_edge_index=save_edge_index, 
            **kwargs
        )

    def get_policy_loss_and_temperature_loss(
        self, obs_t, extra_graph_level_feats=None
    ):
        self.Q1.requires_grad_(False)
        self.Q2.requires_grad_(False)
        self.Q1.eval()
        self.Q2.eval()
        self.policy.train()

        # get policy;
        policy_density, node_embeds, graph_embeds = self.policy(
            obs_t, extra_graph_level_feats, get_graph_embeds=True
        )

        if self.UT_trick:
            # currently works for gaussian policy;
            if self.with_entropy:
                log_pi_integral = -policy_density.entropy().sum(-1)
            else:
                log_pi_integral = policy_density.log_prob_UT_trick().sum(-1)
            UT_trick_samples = policy_density.get_UT_trick_input()
            
            # In GraphOpt there is only one encoder in the policy;
            # so Q-funcs use the embeds from it;
            temp_embeds = graph_embeds.detach()
            q1_integral = batch_UT_trick_from_samples(
                self.Q1.net, temp_embeds, UT_trick_samples
            )
            q2_integral = batch_UT_trick_from_samples(
                self.Q2.net, temp_embeds, UT_trick_samples
            )
            q_integral = torch.min(q1_integral, q2_integral).view(-1)

            # get policy_loss;
            self.policy_loss = (
                np.exp(self.log_temperature.item()) * log_pi_integral
                - q_integral
            ).mean()

            # get temperature loss;
            if self.fixed_temperature is None:
                self.temperature_loss = -(
                    self.log_temperature.exp()
                    * (log_pi_integral + self.entropy_lb).detach()
                ).mean()
        else:
            # do reparam trick;
            if self.use_valid_samples:
                repr_trick1, repr_trick2 = policy_density.rsample(k_proposals=10)
                repr_trick1, repr_trick2 = get_valid_proposal(node_embeds.detach(), obs_t, repr_trick1, repr_trick2)
            else:
                repr_trick1, repr_trick2 = policy_density.rsample()
            assert repr_trick1.requires_grad and repr_trick2.requires_grad

            # get log prob for policy optimisation;
            if self.with_entropy:
                log_prob = -policy_density.entropy().sum(-1)
            else:
                log_prob = policy_density.log_prob(
                    repr_trick1, repr_trick2
                ).sum(-1)

            obs_t = graph_embeds.detach()  # pretend this was input with no grad tracking;
            qfunc_in = torch.cat((obs_t, repr_trick1, repr_trick2), -1)
            q_est = torch.min(
                self.Q1.net(qfunc_in), self.Q2.net(qfunc_in)
            ).view(-1)

            # get loss for policy;
            self.policy_loss = (
                math.exp(self.log_temperature.item()) * log_prob - q_est
            ).mean()

        if self.fixed_temperature is None:
            # get temperature loss;
            self.temperature_loss = -(
                self.log_temperature.exp()
                * (log_prob + self.entropy_lb).detach()
            ).mean()
    
        # housekeeping;
        self.policy_losses.append(self.policy_loss.item())
        if self.fixed_temperature is None:
            self.temperature_losses.append(self.temperature_loss.item())
        self.policy.eval()
    
    def update_parameters(self):
        flag = self.multitask_net is not None
        if flag:
            self.optim_multitask_net.zero_grad()
        # update policy params;
        self.policy_optim.zero_grad()
        self.policy_loss.backward()
        # extra grads flow to policy's encoder
        # and multitas net;
        if flag:
            self.gnn_policy_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.policy_optim.step()

        # update temperature;
        if self.fixed_temperature is None:
            self.temperature_optim.zero_grad()
            self.temperature_loss.backward()
            self.temperature_optim.step()

            # add new temperature to list;
            self.temperatures.append(math.exp(self.log_temperature.item()))

        # update qfunc1
        self.Q1_optim.zero_grad()
        self.Q1_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.Q1_optim.step()

        # update qfunc2
        self.Q2_optim.zero_grad()
        self.Q2_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.Q2_optim.step()

        # do grad step for multitask net;
        if flag:
            self.optim_multitask_net.step()
    
    def get_multitask_loss_from_graph_lvl(
        self, graph_lvl_embeds: list, targets
    ):
        self.gnn_policy_loss = self.multitask_coef * nn.MSELoss()(
            self.multitask_net(graph_lvl_embeds[0]).view(-1), 
            targets.view(-1)
        )
    
    def get_q_losses(
        self, obs_t, action_t, reward_t, obs_tp1, 
        terminated_tp1, extra_graph_feats_t=None, 
        extra_graph_feats_tp1=None
    ):
        self.Q1.requires_grad_(True)
        self.Q2.requires_grad_(True)
        self.Q1.train()
        self.Q2.train()
        self.policy.eval()

        # this is for the multitask gnn training;
        graph_lvl_embeds = []
        return_graph_lvl = False
        targets = None

        # this is for the multitask gnn training;
        if self.multitask_net is not None:
            targets = tgnn.global_add_pool(
                obs_t.x[:, -1].view(-1, 1), obs_t.batch
            )
            return_graph_lvl = True

        policy_density, node_embeds, graph_embeds = self.policy(
            obs_tp1, extra_graph_feats_tp1, get_graph_embeds=True
        )
        node_embeds = node_embeds.detach()  # use embeds only as inputs;
        
        # get input for qfunc;
        if self.buffer.action_is_index:
            action_t = get_action_vector_from_idx(
                node_embeds, action_t, graph_embeds.shape[0]
            )
        obs_action_t = torch.cat(
            (
                graph_embeds.detach(), action_t
            ), -1
        )

        # add embeds from policy (not detached);
        # in the graphopt case there is only one encoder
        # - the one from the policy; 
        if self.multitask_net is not None:
            graph_lvl_embeds.append(graph_embeds)
            self.get_multitask_loss_from_graph_lvl(graph_lvl_embeds, targets)

        # the action_is_index boolean will only be considered
        # if self.encoder is not None;
        q1_est = self.Q1.net(obs_action_t).view(-1)
        q2_est = self.Q2.net(obs_action_t).view(-1)

        if self.UT_trick:
            # get (B, 2 * action_dim + 1, action_dim) samples;
            UT_trick_samples = policy_density.get_UT_trick_input()

            # eval expectation of q-target functions by averaging over the
            # 2 * action_dim + 1 samples and get (B, 1) output;
            temp_embeds = graph_embeds.detach()
            qt1_est = batch_UT_trick_from_samples(
                self.Q1t.net, temp_embeds, UT_trick_samples
            )
            qt2_est = batch_UT_trick_from_samples(
                self.Q2t.net, temp_embeds, UT_trick_samples
            )

            # get negative entropy by using the UT trick;
            if self.with_entropy:
                log_probs = -policy_density.entropy().sum(-1)
            else:
                log_probs = policy_density.log_prob_UT_trick().sum(-1)
        else:
            if self.use_valid_samples:
                a1s, a2s = policy_density.sample(k_proposals=10)
                action_tp1 = get_valid_proposal(node_embeds.detach(), obs_tp1, a1s, a2s)
            else:
                # sample future action;
                action_tp1 = policy_density.sample()

            # get log probs;
            if self.with_entropy:
                log_probs = -policy_density.entropy().sum(-1).view(-1)
            else:
                log_probs = (
                    policy_density.log_prob(*action_tp1).sum(-1).view(-1)
                )

            # input for target nets;
            obs_tp1 = graph_embeds.detach()
            # action_tp1 = (a1, a2) -> tuple of action vectors;
            obs_action_tp1 = torch.cat((obs_tp1,) + action_tp1, -1)
            qt1_est, qt2_est = self.Q1t.net(obs_action_tp1), self.Q2t.net(
                obs_action_tp1
            )

        # use the values from the target net that
        # had lower value predictions;
        q_target = (
            torch.min(qt1_est, qt2_est).view(-1)
            - self.log_temperature.exp() * log_probs
        )

        q_target = (
            reward_t + (1 - terminated_tp1.int()) * self.discount * q_target
        ).detach()

        # loss for first q func;
        self.Q1_loss = nn.MSELoss()(q1_est, q_target)

        # loss for second q func;
        self.Q2_loss = nn.MSELoss()(q2_est, q_target)

        self.Q1_losses.append(self.Q1_loss.item())
        self.Q2_losses.append(self.Q2_loss.item())
    
    def train_one_epoch(self):
        self.policy.eval()
        # presample if needed;
        self.check_presample()

        for _ in tqdm(range(self.num_iters)):
            # sample paths;
            self.policy.eval()
            self.buffer.collect_path(
                self.env,
                self,
                self.num_steps_to_sample,
            )

            # sample paths with delta func policy;
            r, _, code, ep_len, _, obs, _, _, _ = self.buffer.get_single_ep_rewards_and_weights(
                self.env,
                self,
                reward_encoder=self.old_encoder
            )
            if self.buffer.per_decision_imp_sample:
                r = r.sum().item()
            else:
                r = r.item()
            self.eval_path_returns.append(r)
            assert ep_len == obs.edge_index.shape[-1] // 2 - self.env.num_edges_start_from
            self.eval_path_lens.append(ep_len)

            if self.cache_best_policy:
                if self.best_eval_return is None or self.eval_path_returns[-1] > self.best_eval_return:
                    self.best_eval_return = self.eval_path_returns[-1]
                    self._cache()

            # do the gradient updates;
            for _ in range(self.num_grad_steps):
                (
                    obs_t,
                    action_t,
                    reward_t,
                    obs_tp1,
                    terminated_tp1,
                ) = self.buffer.sample(self.batch_size)
                extra_graph_level_feats_t = None
                extra_graph_level_feats_tp1 = None
                if self.buffer.transform_ is not None:
                    if self.buffer.transform_.get_graph_level_feats_fn is not None:
                        assert self.buffer.transform_.n_extra_cols_append > 0
                        extra_graph_level_feats_t = self.buffer.transform_.get_graph_level_feats_fn(obs_t)
                        extra_graph_level_feats_tp1 = self.buffer.transform_.get_graph_level_feats_fn(obs_tp1)

                assert reward_t is None
                if self.buffer.state_reward:
                    r_in = self.old_encoder(obs_tp1)[0]
                    reward_t = self.env.reward_fn(
                        r_in
                    ).detach().view(-1) * self.buffer.reward_scale
                else:
                    raise NotImplementedError(
                        'GraphOpt only uses state reward func!'
                    )
                    reward_t = self.env.reward_fn(
                        (obs_t, action_t), extra_graph_level_feats_t, 
                        action_is_index=self.buffer.action_is_index
                    ).detach().view(-1) * self.buffer.reward_scale
                if self.zero_interm_rew:
                    # assume all rewards before termination were 0.
                    reward_t = reward_t * terminated_tp1
                
                # print(action_t.T, reward_t, sep='\n', end='\n\n')
                # get temperature and policy loss;
                self.get_policy_loss_and_temperature_loss(obs_t, extra_graph_level_feats_t)

                # value func updates;
                self.get_q_losses(
                    obs_t,
                    action_t,
                    reward_t,
                    obs_tp1,
                    terminated_tp1,
                    extra_graph_level_feats_t,
                    extra_graph_level_feats_tp1
                )

                # grad step on policy, temperature, Q1 and Q2;
                self.update_parameters()

                # target q funcs update;
                self.track_qfunc_params()
            # after doing the grad steps, change the reward encoder;
            track_params(self.old_encoder, self.policy.encoder, tau=1)
