import sys
import os
path_network = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path_network)
import numpy as np
import torch
import torch.nn as nn
from trainer.combine import actor_net, CriticAdv

class AgentBase:
    def __init__(self, args=None):
        self.learning_rate = 1e-4 if args is None else args['learning_rate']
        self.soft_update_tau = 2 ** -8 if args is None else args['soft_update_tau']  # 5e-3 ~= 2 ** -8
        self.state = None  # set for self.update_buffer(), initialize before training
        self.device = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None
        self.criterion = None
        self.get_obj_critic = None
        self.train_record = {}

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        `int net_dim` the dimension of networks (the width of neural networks)
        `int state_dim` the dimension of state (the number of state vector)
        `int action_dim` the dimension of action (the number of discrete action)
        `bool if_per` Prioritized Experience Replay for sparse reward
        """

    def select_action(self, state) -> np.ndarray:
        """Select actions for exploration

        :array state: state.shape==(state_dim, )
        :return array action: action.shape==(action_dim, ), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0]
        return action.cpu().numpy()

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        """actor explores in env, then stores the env transition to ReplayBuffer

        :env: RL training environment. env.reset() env.step()
        :buffer: Experience Replay Buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explored target_step number of step in env
        :float reward_scale: scale reward, 'reward * reward_scale'
        :float gamma: discount factor, 'mask = 0.0 if done else gamma'
        :return int target_step: collected target_step number of step in env
        """
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        `buffer` Experience replay buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explore target_step number of step in env
        `int batch_size` sample batch_size of data for Stochastic Gradient Descent
        :float repeat_times: the times of sample batch = int(target_step * repeat_times) in off-policy
        :return float obj_a: the objective value of actor
        :return float obj_c: the objective value of critic
        """

    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))

    def update_record(self, **kwargs):
        """update the self.train_record for recording the metrics in training process
        :**kwargs :named arguments is the metrics name, arguments value is the metrics value.
        both of them will be prined and showed in tensorboard
        """
        self.train_record.update(kwargs)

class AgentPPO(AgentBase):
    def __init__(self, args=None):
        super().__init__(args)
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip),
        self.ratio_clip = 0.3 if args is None else args['ratio_clip']
        # could be 0.01 ~ 0.05
        self.lambda_entropy = 0.05 if args is None else args['lambda_entropy']
        # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.lambda_gae_adv = 0.97 if args is None else args['lambda_gae_adv']
        # if use Generalized Advantage Estimation
        self.if_use_gae = True if args is None else args['if_use_gae']
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        self.if_use_dn = False if args is None else args['if_use_dn']
        self.gamma_att = 0.9 if args is None else args['gamma_att']

        self.noise = None
        self.optimizer = None
        self.compute_reward = None  # attribution

    def init(self, InitDict, reward_dim=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv
        self.batch_size = InitDict['batch_size']
        self.learning_rate = InitDict['learning_rate']
        self.cri = CriticAdv().to(self.device)
        if InitDict['if_load_model']:
            state_dict = torch.load(InitDict['critic_path'],map_location=self.device)  # 加载模型
            self.cri.load_state_dict(state_dict)
        self.act = actor_net(InitDict['batch_size'],InitDict['N_step'],InitDict['actor_path'],InitDict['gpu'],InitDict['possion_num'],InitDict['speed_limiter'],InitDict['lenth']).to(self.device)

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.act])
        sys.stderr.write('\nNumber of action net parameters: \t pi: %d'%var_counts)
        var_counts = tuple(count_vars(module) for module in [self.cri])
        sys.stderr.write('\nNumber of critic net parameters: \t v: %d\n'%var_counts)

        self.optimizer = torch.optim.AdamW([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
        self.criterion = torch.nn.SmoothL1Loss()


    @staticmethod
    def select_action(state, policy,device='cuda:0'):
        """select action for PPO

       :array state: state.shape==(state_dim, )

       :return array action: state.shape==(action_dim, )
       :return array noise: noise.shape==(action_dim, ), the noise
       """
        states = torch.as_tensor((state,), dtype=torch.float32).detach_().to(device)

        if states.ndim == 1:
            states.unsqueeze(0)
        action = policy.get_action(states)[0]
        return action.detach().cpu().numpy()

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_state = buffer.sample_all()
            buf_reward = buf_reward.to(self.device)
            buf_mask = buf_mask.to(self.device)
            buf_action = buf_action.to(self.device)
            buf_state = buf_state.to(self.device)


            bs =  self.batch_size  # set a smaller 'bs: batch size' when out of GPU memory.


            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            self.act.reservoir_network.reset()
            buf_logprob = self.act.compute_logprob(buf_state, buf_action).unsqueeze(dim=1)
            self.act.reservoir_network.reset()
            buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for idx in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_ret = buf_r_ret[indices]
            logprob = buf_logprob[indices]
            adv = buf_adv[indices]
            buf_traj = buf_value[indices]

            new_logprob = self.act.compute_logprob(state, action).unsqueeze(dim=1)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            approx_kl = (logprob - new_logprob).mean().item()

            if approx_kl > 0.01:
                sys.stderr.write('Early stopping at step %d due to reaching max kl.'%idx)
                break
            obj_surrogate1 = adv * ratio
            obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy


            value = self.cri(state)  # critic network predicts the reward_sum (Q value) of state
            # clipv的技巧，防止更新前后V差距过大，对其进行惩罚
            value_clip = buf_traj + torch.clamp(value - buf_traj, -self.ratio_clip, self.ratio_clip)
            obj_critic = torch.max(self.criterion(value, r_ret),self.criterion(value_clip, r_ret))
            

            obj_united = obj_actor + obj_critic / (r_ret.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()
            self.act.reservoir_network.reset()
        if idx>0:
            self.update_record(obj_a=obj_surrogate.item(),
                            obj_c=obj_critic.item(),
                            obj_tot=obj_united.item(),
                            kl=approx_kl,
                            a_std=self.act.a_std_log.exp().mean().item(),
                            entropy=(-obj_entropy.item()))
        return self.train_record

    def compute_reward_adv(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]
        buf_adv = buf_r_ret - (buf_mask * buf_value)
        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv

    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        pre_adv = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                              device=self.device)  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]

            buf_adv[i] = buf_reward[i] + buf_mask[i] * pre_adv - buf_value[i]
            pre_adv = buf_value[i] + buf_adv[i] * self.lambda_gae_adv

        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])
