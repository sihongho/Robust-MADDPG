# Robust Multi-Agent Reinforcement Learning Under Training Partners' Policy Uncertainty

This is the code for implementing the Robust MADDPG (rmaddpg) algorithm. 
The code is modified from https://github.com/openai/maddpg and https://github.com/dadadidodi/m3ddpg .

For Multi-Agent Particle Environments (MPE) installation, please refer to https://github.com/openai/multiagent-particle-envs

For new scenarios we use in our paper, please refer to https://github.com/SihongHo/multiagent-particle-envs

- To run the code, `cd` into the `experiments` directory and run `train.py`:

``python train.py --scenario simple``

- You can replace `simple` with any environment in the MPE you'd like to run.

### Command-line options

#### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"simple"`)

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `60000`)

- `--num-adversaries`: number of adversaries in the environment (default: `0`)

- `--good-policy`: algorithm used for the 'good' (non adversary) policies in the environment
(default: `"rmaddpg"`; options: {`"rmaddpg"`, `"maddpg"`, `"ddpg"`})

- `--adv-policy`: algorithm used for the adversary policies in the environment
(default: `"rmaddpg"`; options: {`"rmaddpg"`, `"maddpg"`, `"ddpg"`})

#### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

- `--partner-type`: partner agents type (default: `888`, means all partner agents are indifference). 


In a n-agents environment, it should be a string with length `n*n` consists of numbers `0`, `1`, `2`, `3`, `4`. 


The numbers denote the partner agents' type: `0: indifferent`, `1: cooperate`, `2: compete`, `3: assist`, `4: attack`.


The string can be formulated as a n*n symmetric Hollow matrix (whose diagonal elements are all equal to zero) from left to right, from top to bottom.


For example, in a 2-agents environment, if the input is `"0110"`, then the 2*2 matrix is 

                               0 1
                               
                               1 0
                               
which means: agent 1 and agent 2 are cooperators.

- `--uncertainty-type`: uncertainty type (default: `0`; options: {`1`, `2`, `3`}). `0: none`, `1:reward`, `2:action`, `3:observation`

- `--uncertainty-std`: uncertainty level (default: `1.0`; options: {`1.0`, `2.0`, `3.0`, ...}).


