# Inverse Reinforcement Learning of Graph Objective Functions:
This work investigates the effect of per-decision importance 
samples on the reward function-learning objective as compared 
to the default per-episode importance samples. 
As a baseline, this method was compared to GraphOpt 
[GraphOpt](https://arxiv.org/abs/2007.03619).

I also test 
the addition of a separate GNN for the reward, rahter than using 
only a single GNN, jointly trained with the policy, as done in 
GraphOpt. The addition of the reward-specific GNN allows to do 
full reward function transfer rather than only transferring the 
MLP part of the reward. Based on experiments, it appears 
this addition helps to learn new policy that constructs graphs
more similar in topology to the target graph compared to our 
implementation of GraphOpt.

Disclaimer: at the time of this work I could not find any implementation of GraphOpt in any online publicly-available 
repository related to the authors of
[GraphOpt](https://arxiv.org/abs/2007.03619). To that end, I 
implemented my own version based on the description in their paper.


## IRL procedure is based on [Guided Cost Learning - C. Finn paper](https://arxiv.org/abs/1603.00448); and [GraphOpt](https://arxiv.org/abs/2007.03619);
* The contribution is the addition of per-decision importance 
samples rather than per-episode ones. 
    * That way each reward 
is weighted by the importance sample from the 
path up to that point in 
time, rather than the entire path.
    * Intuitively, we may have a very likely path up to time $t$, 
    but then a very unlikely continuation up to time $T$. If 
    a per-episode importance weight is used, all rewards (even 
    from the part where we had a likely path) get downgraded 
    weights because of the unlikely trajectory sapmled after 
    time $t$.
    * On the other hand, per-decision samples, more accurately 
    attribute weights based on the path up to observing the 
    current reward.
* The sampling of the paths is done from a policy trained with 
the current configuration of the reward $r_\psi$, using the 
Soft Actor-Critic algorithm (MaxEnt RL algo), [SAC-paper](https://arxiv.org/abs/1812.05905).

## Model Architectures;

### GraphOpt:
![alt text](https://github.com/mariovas3/urban-nets-style-transfer/blob/master/ucl-pg-project-latex-template/GO_schematic2.png)

### Mine:
![alt text](https://github.com/mariovas3/urban-nets-style-transfer/blob/master/ucl-pg-project-latex-template/my_model_architecture2.png)

## Side docs for mujoco - used to test policy learners:
* `pip install -U portalocker`
* `pip install -U lockfile`
* `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<your_name>/.mujoco/mjpro150/bin  # put this in the ~/.bashrc and source the file;`
* `sudo apt-get install libosmesa6-dev  # fix the missing GL/osmesa.h file error;`
* `sudo apt-get install patchelf  # fix no such file patchelf error;`
* Provided you have downloaded mjpro150 and have an access key the following should install `mujoco-py`:
    `pip install -U 'mujoco-py<1.50.2,>=1.50.1'`

## Experiments recipes:
### Barabasi graphs:
* GraphOpt with per-decision imporance samples:
`python tests/test_irl_ba20.py net_hiddens=256,256 encoder_hiddens=64,64 embed_dim=32 max_size=100000 per_decision_imp_sample=1 quad_reward_penalty_coef=0.1 with_mlp_batch_norm=0 with_mlp_layer_norm=1 do_graphopt=1 no_q_encoder=1 with_mono=0 with_lcr=0 log_sigma_min=-20 log_sigma_max=2 reward_lr=3e-3 do_dfs_expert_paths=0 num_iters=10 num_grad_steps=10 ortho_init=0 unnorm_policy=1 use_valid_samples=1 seed=0`
* GraphOpt using per-episode importance samples:
`python tests/test_irl_ba20.py net_hiddens=256,256 encoder_hiddens=64,64 embed_dim=32 max_size=100000 per_decision_imp_sample=0 quad_reward_penalty_coef=0.1 with_mlp_batch_norm=0 with_mlp_layer_norm=1 do_graphopt=1 no_q_encoder=1 with_mono=0 with_lcr=0 log_sigma_min=-20 log_sigma_max=2 reward_lr=3e-3 do_dfs_expert_paths=0 num_iters=10 num_grad_steps=10 ortho_init=0 unnorm_policy=1 use_valid_samples=1 seed=0`
* Sep reward encoder using per-decision importance samples:
`python tests/test_irl_ba20.py net_hiddens=256,256 encoder_hiddens=64,64 embed_dim=32 max_size=100000 per_decision_imp_sample=1 quad_reward_penalty_coef=0.1 with_mlp_batch_norm=0 with_mlp_layer_norm=1 do_graphopt=0 no_q_encoder=1 with_mono=0 with_lcr=0 log_sigma_min=-20 log_sigma_max=2 reward_lr=3e-3 do_dfs_expert_paths=0 num_iters=10 num_grad_steps=10 ortho_init=0 unnorm_policy=1 use_valid_samples=1 seed=0`
* Sep reward encoder using per-episode importance samples:
`python tests/test_irl_ba20.py net_hiddens=256,256 encoder_hiddens=64,64 embed_dim=32 max_size=100000 per_decision_imp_sample=0 quad_reward_penalty_coef=0.1 with_mlp_batch_norm=0 with_mlp_layer_norm=1 do_graphopt=0 no_q_encoder=1 with_mono=0 with_lcr=0 log_sigma_min=-20 log_sigma_max=2 reward_lr=3e-3 do_dfs_expert_paths=0 num_iters=10 num_grad_steps=10 ortho_init=0 unnorm_policy=1 use_valid_samples=1 seed=0`
* For using the Unscented Transform trick, add the `UT_trick=1` flag; 
the default is `UT_trick=0` which falls back to reparameterisation 
trick for estimating expectations. In practice, didn't see 
any improvement when using UT trick.

### The scripts above were run on a remote server;
* After the relevant output from the above scripts was saved, 
the results were analysed by the `experiments_analysis.ipynb` 
notebook in the `notebooks` directory.