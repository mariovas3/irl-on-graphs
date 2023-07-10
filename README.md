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