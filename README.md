# Urban Nets Style Transfer

## Side docs for mujoco - used to test policy learners:
* `pip install -U portalocker`
* `pip install -U lockfile`
* `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<your_name>/.mujoco/mjpro150/bin  # put this in the ~/.bashrc and source the file;`
* `sudo apt-get install libosmesa6-dev  # fix the missing GL/osmesa.h file error;`
* `sudo apt-get install patchelf  # fix no such file patchelf error;`
* Provided you have downloaded mjpro150 and have an access key the following should install `mujoco-py`:
    `pip install -U 'mujoco-py<1.50.2,>=1.50.1'`

## Policy learners:
### Soft Actor-Critic:
* [SAC-paper](https://arxiv.org/abs/1812.05905)
* I have been testing this on the MuJoCo envs - primarily `Hopper-v2` and `Ant-v2`. I have found that SAC benefits a lot from UT when the action space is large (the case for Ant - 8 dim action space, 27 dim obs space). There is little difference in the setting with small action space (action dim is 3 for Hopper) relative to the reparam trick as given in the paper by the authors.
* UT might prove helpful for eval of expectations in high dim action spaces. In the case of diag Gauss policies, examples of eig vecs of the cov matrix are axis-aligned unit vectors, with eig vals - the componentwise variances in the Gauss vector. This makes it convenient to get UT input by adding and subtracting axis-aligned unit vectors scaled by the relevant standard deviations (sqrt of eig vals) from the mean vector - giving 2D+1 inputs, where D is the length of the Gauss vector (usually the action dim).

### Graph policy:
* Stochastic policy $\pi(a|graph)$ that samples vector $a$ and finds 1-NN in gnn embedding space (KDTree implementation). Then calculate cos distance with other node GNN embeddings and concat this vector to GNN embeddings and do another MLP that maps $NN^{(2)}: \mathbb{R}^{emb\_dim+1}\rightarrow \mathbb{R}$ and take softmax to spit out a node index to connect to:
    * $Z = GNN(graph)\in \mathbb{R}^{num\_nodes\times emb\_dim}$.
    * $a \sim Dist(NN^{(1)}(Z))$.
    * $z^{(1)}= \arg \min_{z\in Z} d^{(1)}(a, Z).$
    * $feat=d^{(2)}(z^{(1)}, Z)\in \mathbb{R}^{num\_nodes}$.
    * $\tilde{Z}:=(Z, feat)\in \mathbb{R}^{num\_nodes\times emb\_dim+1}$.
    * $z^{(2)}:=\arg \max_{z\in Z} softmax(NN^{(2)}(\tilde{Z}))$.
    * Connect $z^{(1)}$ and $z^{(2)}$ with edge.

* There should be some budgeting there for allowing some edges and not others.
* The policy, the reward function and the value functions should have their own GNN encoders. For the reward this is necessary since we don't want to let the policy gradients implicitly change the reward function (would be the case if the GNN was shared by the policy and the reward function).
* For the sake of compatibility with the openai gym Env, the action in this graph setting will be $\tilde{a}=(a, Z)$.

### Graph  Buffer:
* The observations should be instances of `torch_geometric.data.Data`. The actions should be tuples `(idx1, idx2)` of length 2 containing the indexes of the nodes to be connected. Based on these indexes, `x_embeds[idx1], x_embeds[idx2]` should be passed to the `log_prob` method of the `TwoStageGaussDist` instance. The `x_embeds` will be computed anew for each batch during the gradient steps. The state will be the graph itself rather than its embedding from the GNN to allow training of the GNN, therefore. And the actions' interpretation is that we emitted vectors close to the embeddings of the selected vectors.
    * Given this setup, I might need a new Graph Buffer class, to accommodate for these formats.
* The alternative would be to store `(a1, a2)` - a tuple of two vectors for for the actions from when the path was sampled (some point in the past). These, however, would have been output based on a different encoding from the GNN, so are not in general guaranteed to map to the same node embeddings given the new parameters of the GNN.
* Another alternative would be to store GNN embeddings as observations and then generate a gaussian based on the new params of the policy. This, however, does not train the GNN since we don't use it if we store embeddings for the observations.