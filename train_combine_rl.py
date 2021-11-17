# -*- coding: utf-8 -*-
"""
@author:yym
"""
import sys
import os
import ray
import time

path_network = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path_network)

import numpy as np
import numpy.random as rd
import torch
from trainer.agent import AgentPPO
from utils.simulation import simulate
import datetime
from utils.buffer import ReplayBuffer, ReplayBufferMP
from utils.num_in_out import get_data, get_data_prefrontal
from utils.evaluate import RecordEpisode, RecordEvaluate, Evaluator

class Arguments:
    def __init__(self, configs):
        self.configs = configs
        self.gpu_id = configs['gpu_id']  # choose the GPU for running. gpu_id is None means set it automatically
        # current work directory. cwd is None means set it automatically
        self.cwd = configs['cwd'] if 'cwd' in configs.keys() else None
        # current work directory with time.
        self.if_cwd_time = configs['if_cwd_time'] if 'cwd' in configs.keys() else False
        self.expconfig = configs['expconfig']
        # initialize random seed in self.init_before_training()

        self.random_seed = configs['random_seed']

        # Deep Reinforcement Learning algorithm
        self.agent = configs['agent']
        self.agent['agent_name'] = self.agent['class_name']().__class__.__name__
        self.trainer = configs['trainer']
        self.interactor = configs['interactor']
        self.buffer = configs['buffer']
        self.evaluator = configs['evaluator']
        self.InitDict = configs['InitDict']

        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)

        '''if_per_explore'''
        if self.buffer['if_on_policy']:
            self.if_per_explore = False
        else:
            self.if_per_explore = True

    def init_before_training(self, if_main=True):
        '''set gpu_id automatically'''
        if self.gpu_id is None:  # set gpu_id automatically
            import sys
            self.gpu_id = sys.argv[-1][-4]
        else:
            self.gpu_id = self.gpu_id
        # if not self.gpu_id.isdigit():  # set gpu_id as '0' in default
        #     self.gpu_id = '0'

        '''set cwd automatically'''
        if self.cwd is None:
            if self.if_cwd_time:
                curr_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            else:
                curr_time = 'current'
            if self.expconfig is None:
                self.cwd = f'./logs/{self.agent["agent_name"]}/exp_{curr_time}_cuda:{self.gpu_id}'
            else:
                self.cwd = f'./logs/{self.agent["agent_name"]}/exp_{curr_time}_cuda:{self.gpu_id}_{self.expconfig}'

        if if_main:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)
            fw = open(self.cwd+"/config.txt",'w+') 
            fw.write(str(self.configs)) #transfer dict to str 
            fw.close()

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

def make_env(InitDict, id=None):


    env = simulate(0,0,0,0,0,InitDict['lenth'])

    global observation_dim
    global action_dim
    action_dim = env.action_space
    observation_dim = env.observation_space
    
    return env

@ray.remote
class InterActor(object):

    def __init__(self, id, args):
        self.id = id
        args.init_before_training(if_main=False)
        self.env = make_env(args.InitDict, self.id)
        self.env_max_step = args.InitDict['max_step']
        global observation_dim
        global action_dim
        observation_dim = self.env.observation_space
        action_dim = self.env.action_space   
        self.reward_scale = args.interactor['reward_scale']
        self._horizon_step = args.interactor['horizon_step'] // args.interactor['rollout_num']
        self.gamma = args.interactor['gamma'] if type(args.interactor['gamma']) is np.ndarray else np.ones(
            args.InitDict['reward_dim']) * args.interactor['gamma']
        self.action_dim = action_dim

        self.buffer = ReplayBuffer(
            max_len=args.buffer['max_buf'] // args.interactor['rollout_num'] + args.InitDict['max_step'],
            if_on_policy=args.buffer['if_on_policy'],
            state_dim=observation_dim,
            action_dim= action_dim,
            reward_dim=args.InitDict['reward_dim'],
            if_per=False,
            if_gpu=False)
        
        self.record_episode = RecordEpisode()

    @ray.method(num_returns=1)
    def explore_env(self, select_action, policy,device):
        self.buffer.empty_buffer_before_explore()
        actual_step = 0
        actual_traj = 0
        while actual_step < self._horizon_step:
            state = self.env.reset()
            for i in range(self.env_max_step):
                action = select_action(state, policy,device)
                next_s, reward, done, _ = self.env.step(action)
                done = True if i == (self.env_max_step - 1) else done
                self.buffer.append_buffer(state,
                                          action,
                                          reward * self.reward_scale,
                                          np.zeros(self.gamma.shape) if done else self.gamma)
                actual_step += 1
                if done:
                    actual_traj+=1
                    break
                state = next_s
        self.buffer.update_now_len_before_sample()
        return actual_traj, \
               self.buffer.buf_state[:self.buffer.now_len], \
               self.buffer.buf_action[:self.buffer.now_len], \
               self.buffer.buf_reward[:self.buffer.now_len], \
               self.buffer.buf_gamma[:self.buffer.now_len]

    def exploite_env(self, select_action, policy, eval_times,device):
        self.record_episode.clear()
        eval_record = RecordEvaluate()

        for _ in range(eval_times):
            state = self.env.reset()
            for _ in range(self.env_max_step):
                action = select_action(state, policy,device)
                next_s, reward, done, info = self.env.step(action)
                self.record_episode.add_record(reward, info)
                if done:
                    break
                state = next_s
            eval_record.add(self.record_episode.get_result())
            self.record_episode.clear()
        return eval_record.results


class Trainer(object):

    def __init__(self, args_trainer, agent, buffer,device='cuda:0'):
        self.device = device
        self.agent = agent
        self.buffer = buffer
        self.sample_step = args_trainer['sample_step']
        self.batch_size = args_trainer['batch_size']
        self.policy_reuse = args_trainer['policy_reuse']

    def train(self):
        self.agent.act.to(self.device)
        self.agent.cri.to(self.device)
        train_record = self.agent.update_net(self.buffer, self.sample_step, self.batch_size, self.policy_reuse)
        if self.buffer.if_on_policy:
            self.buffer.empty_buffer_before_explore()
        return train_record

observation_dim = 0
action_dim = 0

def beginer(config, params=None):
    args = Arguments(config)
    args.init_before_training()
    args_id = ray.put(args)
    #######Init######

    interactors = [InterActor.remote(i, args_id) for i in range(args.interactor['rollout_num'])]
    print('state dim',observation_dim)
    make_env(args.InitDict)
    args.InitDict['state_dim'] = observation_dim
    args.InitDict['action_dim'] = action_dim
    print('state dim',observation_dim)
    agent = args.agent['class_name'](args.agent)
    agent.init(InitDict=args.InitDict,
               reward_dim=args.InitDict['reward_dim'],)
    buffer_mp = ReplayBufferMP(
        max_len=args.buffer['max_buf'] + args.InitDict['max_step'] * args.interactor['rollout_num'],
        state_dim=observation_dim,
        action_dim= action_dim,
        reward_dim=args.InitDict['reward_dim'],
        if_on_policy=args.buffer['if_on_policy'],
        if_per=args.buffer['if_per'],
        rollout_num=args.interactor['rollout_num'])
    trainer = Trainer(args.trainer, agent, buffer_mp)
    evaluator = Evaluator(args)
    rollout_num = args.interactor['rollout_num']

    #######Interacting Begining#######
    start_time = time.time()
    device = 'cuda'
    policy_id = ray.put(agent.act.to(device))
    
    while (evaluator.record_totalstep < evaluator.break_step) or (evaluator.record_satisfy_reward):
        #######Explore Environment#######
        episodes_ids = [interactors[i].explore_env.remote(agent.select_action, policy_id,device) for i in
                        range(rollout_num)]
        assert len(episodes_ids) > 0
        sample_step = 0
        for i in range(len(episodes_ids)):
            done_id, episodes_ids = ray.wait(episodes_ids)
            actual_step, buf_state, buf_action, buf_reward, buf_gamma = ray.get(done_id[0])
            sample_step += actual_step
            buffer_mp.extend_buffer(buf_state, buf_action, buf_reward, buf_gamma, i)
        evaluator.update_totalstep(sample_step)
        #######Training#######
        trian_record = trainer.train()
        evaluator.tb_train(trian_record)
        #######Evaluate#######
        device = 'cuda'
        policy_id = ray.put(agent.act.to(device))
        evalRecorder = RecordEvaluate()
        if_eval = True
        #######pre-eval#######
        
        if evaluator.pre_eval_times > 0:
            eval_results = ray.get(
                [interactors[i].exploite_env.remote(agent.select_action, policy_id, eval_times=evaluator.pre_eval_times, device=device) for i in
                 range(rollout_num)])
            for eval_result in eval_results:
                evalRecorder.add_many(eval_result)
            eval_record = evalRecorder.eval_result()
            if eval_record['reward'][0]['max'] < evaluator.target_reward:
                if_eval = False
                evaluator.tb_eval(eval_record)
        #######eval#######
        if if_eval:
            eval_results = ray.get(
                [interactors[i].exploite_env.remote(agent.select_action, policy_id, eval_times=(evaluator.eval_times),device=device)
                 for i in range(rollout_num)])
            for eval_result in eval_results:
                evalRecorder.add_many(eval_result)
            eval_record = evalRecorder.eval_result()
            evaluator.tb_eval(eval_record)
        #######Save Model#######
        evaluator.analyze_result(eval_record)
        evaluator.iter_print(trian_record, eval_record, use_time=(time.time() - start_time))
        evaluator.save_model(agent.act, agent.cri)
        start_time = time.time()

    print(f'#######Experiment Finished!\t TotalTime:{evaluator.total_time:8.0f}s #######')

max_step = 300
rollout_num = 4
config_ppo = {
    'gpu_id': 0,
    'cwd': None,
    'if_cwd_time': True,
    'expconfig': None,
    'random_seed': 0,
    'agent': {
        'class_name': AgentPPO,
        'net_dim': 128,
        'ratio_clip': 0.3,
        'lambda_entropy': 0.04,
        'lambda_gae_adv': 0.97,
        'if_use_gae': True,
        'if_use_dn': False,
        'learning_rate': 1e-4,
        'soft_update_tau': 2 ** -8,
        'gamma_att' : 0.85,
    },
    'trainer': {
        'batch_size': 256,
        'policy_reuse': 2 ** 4,
        'sample_step': max_step * rollout_num,
    },
    'interactor': {
        'horizon_step': max_step * rollout_num,
        'reward_scale': 2 ** 0,
        'gamma': 0.99,
        'rollout_num': rollout_num,
    },
    'buffer': {
        'max_buf': max_step * rollout_num,
        'if_on_policy': True,
        'if_per': False,
    },
    'evaluator': {
        'pre_eval_times': 2,  # for every rollout_worker 0 means cencle pre_eval
        'eval_times': 4,  # for every rollout_worker
        'if_save_model': True,
        'break_step': 2e6,
        'satisfy_reward_stop': False,
    },
    'InitDict':{
        'target_reward':100,
        'reward_dim': 1,
        'lenth': 2000,
        'max_step': 250,
        'state_dim': None,
        'mid_dim':128,
        'block_size': 9,
        'block_size_state':9,
        'batch_size': 256,
        'learning_rate': 1e-4,
        'if_load_model': False,
        'N_step': 2,
        'actor_path': None,
        'critic_path': None,
        'gpu': 0,
        'possion_num': 50,
        'speed_limiter': 100,

    },
}

def demo_ppo():

    config_ppo['agent']['lambda_entropy'] = 0.05
    config_ppo['agent']['lambda_gae_adv'] = 0.97
    config_ppo['interactor']['rollout_num'] = 16
    config_ppo['agent']['learning_rate'] = 0.2e-4
    config_ppo['InitDict']['learning_rate'] = config_ppo['agent']['learning_rate']
    config_ppo['trainer']['batch_size'] = 256
    config_ppo['trainer']['sample_step'] = 1024
    config_ppo['InitDict']['batch_size'] = config_ppo['trainer']['batch_size']
    config_ppo['interactor']['horizon_step'] = config_ppo['trainer']['sample_step']
    config_ppo['trainer']['policy_reuse'] = 8
    config_ppo['interactor']['gamma'] = 0.99
    config_ppo['evaluator']['break_step'] = int(2e5)
    config_ppo['buffer']['max_buf'] = config_ppo['interactor']['horizon_step']
    config_ppo['gpu_id'] = '0'
    config_ppo['InitDict']['gpu'] = config_ppo['gpu_id']
    config_ppo['if_cwd_time'] = True
    config_ppo['expconfig'] = 'RLtest'
    config_ppo['random_seed'] = 59
    beginer(config_ppo)



if __name__ == '__main__':
    from trainer.agent import *
    # ray.init()
    ray.init(local_mode=True)

    demo_ppo()