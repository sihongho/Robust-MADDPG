import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U
from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])
    
# to-do: p_train function for partner-agents
def partner_p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, adversarial, adv_eps, adv_eps_s, num_adversaries, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="ptrainer", reuse=None):
    # Policy network
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="par_action"+str(i)) for i in range(len(act_space_n))]

        p_index = p_index % 10
        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="par_p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("par_p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()

        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="par_q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)
        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])# train is a function to minimize loss
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)# act is a function from obs to action
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="par_target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("par_target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'par_p_values': p_values, 'par_target_act': target_act}
    
# to-do: q_train function for partner-agents
def partner_q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, adversarial, adv_eps, adv_eps_s, num_adversaries, grad_norm_clipping=None, local_q_func=False, scope="ptrainer", reuse=None, num_units=64):
    # Q (Critic) Network
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="par_action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="par_target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        q_index = q_index % 10
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="par_q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("par_q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="par_target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("par_target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'par_q_values': q_values, 'par_target_q_values': target_q_values}

# to-do: new class PartnerAgentTrainer
class PartnerAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func, policy_name, adversarial, noise_type, noise_std):
        self.name = name # name of a agent: type = string
        self.scope = self.name + "_" + policy_name
        self.n = len(obs_shape_n) # len of obs: type = int; different agents' obs are different
        self.agent_index = agent_index # type = int; scope = n*n
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            # Creates a placeholder for a batch of tensors of a given shape
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="par_observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = partner_q_train(
            scope=self.scope,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            adversarial = adversarial,
            adv_eps = args.adv_eps,
            adv_eps_s = args.adv_eps_s,
            num_adversaries = args.num_adversaries,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = partner_p_train(
            scope=self.scope,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            adversarial = adversarial,
            adv_eps = args.adv_eps,
            adv_eps_s = args.adv_eps_s,
            num_adversaries = args.num_adversaries,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        # Each agent has its own replay buffer
        self.replay_buffer = ReplayBuffer(size = 1e6, noise_type = noise_type, noise_std = noise_std)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        self.policy_name = policy_name
        self.adversarial = adversarial
        self.act_space_n = act_space_n
        self.local_q_func = local_q_func
        
    def debuginfo(self):
        return {'name': self.name, 'index': self.agent_index, 'scope': self.scope,
            'policy_name': self.policy_name, 'adversarial': self.adversarial,
            'local_q_func':self.local_q_func,
            'adv_eps': self.args.adv_eps}

    def action(self, obs):
        # For this agent, given obs, output action
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for a in range(num_sample):
            # to-do: here we replace actions using partners' actions
            target_act_next_n = []
            for j in range(self.n):
                if j == (self.agent_index % 10):
                    target_act_next_n.append(agents[j].p_debug["par_target_act"](obs_next_n[j]))
                else:
                    target_act_next_n.append(agents[j].p_debug["target_act"](obs_next_n[j]))
            target_q_next = self.q_debug['par_target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
