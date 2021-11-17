# -*- coding: utf-8 -*-
"""
@author:yym
Note that batch training is not supported in this version.
"""
import sys
import os

path_network = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path_network)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.NetworkClasses_possion import SNN, EPropRSNU, cere_model, prefrontal_model
from utils.coding_and_decoding import poisson_spike, poisson_spike_multi



class actor_net(nn.Module):
    def __init__(
        self,
        batch_size=1,
        N_step=2,
        load_model=None,
        gpu='0',
        possion_num=50,
        speed_limiter=100,
        lenth=2 * 1000):
        super(actor_net,self).__init__()

        self.batch_size = batch_size
        self.N_step = N_step  # predict step
        self.load_model = load_model
        #dims = (7,7,7)
        dims = (4,4,4)
        self.bias = 4
        self.n_in = dims[0]*dims[1]*dims[2]
        w_mat = 4*20*np.array([[3, 6],[-2, -2]])
        self.steps = 50
        self.ch = 5 #input num
        self.possion_num_snu = possion_num
        self.possion_num = possion_num
        self.speed_limiter = np.ones([1,1])*speed_limiter
        self.encoding_num = 4
        self.dt = 0.1
        self.lenth = np.ones([1,1])*lenth
        self.x_sum = 0
        self.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")

        if load_model == None:
            sys.stderr.write("create new model\n")
            self.reservoir_network = EPropRSNU(self.ch*self.encoding_num,self.n_in)
            self.snu = SNN(batch_size=self.batch_size, input_size=self.n_in, num_classes=self.bias, encoding_num=self.encoding_num, possion_num=self.possion_num, gpu=gpu)

            self.network_x = cere_model(batch_size=self.batch_size,num_in_MF=1,num_out_MF=4,num_out_GC=200,num_out_PC=20,num_out_DCN=1,possion_num=self.possion_num,gpu=gpu)
            self.network_y = cere_model(batch_size=self.batch_size,num_in_MF=1,num_out_MF=4,num_out_GC=200,num_out_PC=20,num_out_DCN=1,possion_num=self.possion_num,gpu=gpu)

            self.network = prefrontal_model(batch_size=self.batch_size,num_hidden1=32,num_hidden2=64,num_hidden3=32,N_step=self.N_step,gpu=gpu)

            # self.network_x = cere_model(batch_size=self.batch_size,num_in_MF=1,num_out_MF=4,num_out_GC=2000,num_out_PC=200,num_out_DCN=1,possion_num=self.possion_num,gpu=gpu)
            # self.network_y = cere_model(batch_size=self.batch_size,num_in_MF=1,num_out_MF=4,num_out_GC=2000,num_out_PC=200,num_out_DCN=1,possion_num=self.possion_num,gpu=gpu)

            # self.network = prefrontal_model(batch_size=self.batch_size,num_hidden1=256,num_hidden2=256,num_hidden3=256,N_step=self.N_step,gpu=gpu)

        else:
            self.load(load_model)

        self.snu = self.snu.float().to(self.device)
        self.network_x = self.network_x.float().to(self.device)
        self.network_y = self.network_y.float().to(self.device)
        self.network = self.network.float().to(self.device)

        #Fix partial model parameters
        self.snu.requires_grad = False

        #self.network_x.requires_grad = False
        #self.network_y.requires_grad = False

        # self.network.fc11.weight.requires_grad = False
        # self.network.fc11.bias.requires_grad = False
        # self.network.fc12.weight.requires_grad = False
        # self.network.fc12.bias.requires_grad = False
        # self.network.fc21.weight.requires_grad = False
        # self.network.fc21.bias.requires_grad = False
        # self.network.fc22.weight.requires_grad = False
        # self.network.fc22.bias.requires_grad = False
        f = np.ones([1,1])*400
        self.belta_true = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=0.5*f/4/2,dim=4)).float().to(self.device)
        self.opts_mx = 0
        self.opts_my = 0
        self.opts_mz = 0

        self.a_std_log = nn.Parameter(torch.zeros(1,3)-0.5 ,requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        
    def forward(self,state):
        '''
        f_x: the force in x axis
        f_y: the force in y axis
        x_z: The distance that the part has traveled
        m_z: Current Z-direction speed of the part
        '''
        batch_size = state.shape[0]
        f_x = state[:,0].reshape(batch_size,1)
        f_y = state[:,1].reshape(batch_size,1)
        x_z = state[:,2].reshape(batch_size,1)
        m_x = state[:,3].reshape(batch_size,1)
        m_y = state[:,4].reshape(batch_size,1)
        m_z = state[:,5].reshape(batch_size,1)

        #init matrix
        self.snu_output_np = np.zeros([batch_size, self.N_step,self.bias])

        self.possion_rate_coding = np.zeros([batch_size, self.n_in,self.possion_num_snu])

        self.m_x = torch.zeros([batch_size, 1]).float().to(self.device)
        self.m_y = torch.zeros([batch_size, 1]).float().to(self.device)
        self.m_dot_z_true = torch.zeros([batch_size, 1]).float().to(self.device)

        # Is it better to use tanh() ?
        f_x_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=(f_x+200)/4,dim=self.encoding_num)).float().to(self.device) #[0,100]
        f_y_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=(f_y+200)/4,dim=self.encoding_num)).float().to(self.device) #[0,100]
        m_x_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=50*(m_x+4)/4,dim=self.encoding_num)).float().to(self.device) #[0,100]
        m_y_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=50*(m_y+4)/4,dim=self.encoding_num)).float().to(self.device) #[0,100]

        cere_out = torch.zeros([state.shape[0], self.N_step, self.bias*4, self.possion_num])

        #init parms for SNU_LSM
        ipts = torch.zeros([batch_size, 5])
        ipts[:, 0] = torch.clamp(m_z.squeeze(),0,100)

        v_h_init = torch.zeros([batch_size,self.n_in]).to(self.device)

        with torch.no_grad():
            for iteration in range(self.N_step):
                
                train_in_spikes = torch.from_numpy(poisson_spike_multi(t=self.possion_num*self.dt,f=ipts, dim=self.encoding_num)).float().to(self.device).reshape(batch_size,-1,self.possion_num)
                # train_in_spikes (1,5,5,50)
                #LSM mode
                rate_coding = self.reservoir_network.forward(train_in_spikes, v_h_init)
        
                rate_coding = rate_coding.permute(1,0,2)
                
                #SNU mode
                self.snu_output = self.snu.forward(input=rate_coding,task="LSM",time_window=self.possion_num)

                #build next step data
                for m in range(4):
                    self.snu_output_np[:,iteration,m] = torch.sum(self.snu_output[:,4*m:4*m+4],dim=1).cpu().detach().numpy()
                ipts,_,__ = np.split(ipts,[1,1],axis = 1) # get delt z
                ipts = np.hstack((ipts,self.snu_output_np[:,iteration])) #t+1 input

                #cere mode
                u_x_coding = self.snu.monitor_fc1[:,0:4,:]
                u_y_coding = self.snu.monitor_fc1[:,4:8,:]
                sigma_x_coding = self.snu.monitor_fc1[:,8:12,:]
                sigma_y_coding = self.snu.monitor_fc1[:,12:16,:]

                sum_x = self.network_x.forward(m_x_coding,u_x_coding,sigma_x_coding,f_x_coding,time_window=self.possion_num)
                sum_y = self.network_y.forward(m_y_coding,u_y_coding,sigma_y_coding,f_y_coding,time_window=self.possion_num)

                sum_mx = sum_x[:,0:4]
                sum_ux = sum_x[:,4:8]
                sum_y = sum_y
                sum_my = sum_y[:,0:4]
                sum_uy = sum_y[:,4:8]

                cere_out[:,iteration,0:4,:] = self.network_x.monitor_DCN_m[:,4:8,:]
                cere_out[:,iteration,4:8,:] = self.network_y.monitor_DCN_m[:,4:8,:]
                cere_out[:,iteration,8:12,:] = sigma_x_coding
                cere_out[:,iteration,12:16,:] = sigma_y_coding

                m_x_coding = self.network_x.monitor_DCN_m[:,0:4,:]
                m_y_coding = self.network_y.monitor_DCN_m[:,0:4,:]

        cere_out = cere_out.to(self.device)
        #pre mode
        u_xt = cere_out[:,:,0:4,:].clone()
        u_xt = u_xt.view(batch_size,-1,self.possion_num)
        u_yt = cere_out[:,:,4:8,:].clone()
        u_yt = u_yt.view(batch_size,-1,self.possion_num)
        sigma_x = cere_out[:,:,8:12,:].clone()
        sigma_x = sigma_x.view(batch_size,-1,self.possion_num)
        sigma_y = cere_out[:,:,12:16,:].clone()
        sigma_y = sigma_y.view(batch_size,-1,self.possion_num)

        lenth_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=self.lenth/20/2,dim=self.encoding_num)).float().to(self.device) #[0，50]
        x_sum_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=x_z/10/2,dim=self.encoding_num)).float().to(self.device) #[0,50]
        m_z_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=m_z/4/2,dim=self.encoding_num)).float().to(self.device) #[0,25]
        
        speed_limiter_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=self.speed_limiter/2,dim=self.encoding_num)).float().to(self.device) #[0,25]
        beltat = self.belta_true.repeat(batch_size,1,1)
        lenth_coding = lenth_coding.repeat(batch_size,1,1)
        speed_limiter_coding = speed_limiter_coding.repeat(batch_size,1,1)

        input11 = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=(f_x+200)/4/2,dim=self.encoding_num)).float().to(self.device)
        input12 = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=(f_y+200)/4/2,dim=self.encoding_num)).float().to(self.device)
        input2 = torch.cat([u_xt,u_yt,sigma_x,sigma_y],1).float().to(self.device) * 0.5
        input3 = torch.cat([lenth_coding,x_sum_coding,m_z_coding],1).float().to(self.device)
        input4 = torch.cat([beltat,m_z_coding,speed_limiter_coding],1).float().to(self.device)
        input = torch.cat([input11,input12,input2,input3,input4],1)


        sum3 = self.network.forward(input,time_window=self.possion_num)

        self.m_dot_z_true = torch.sum(sum3[:,0:4],dim=1)
        # self.belta_true = self.network.monitor_h32[:,4:8,:].view(4,self.possion_num)

        ######################################

        #cere
        u_x_coding_predict = cere_out[:,0,0:4,:]
        u_x_coding_predict = u_x_coding_predict.view(batch_size,-1,self.possion_num)
        u_y_coding_predict = cere_out[:,0,4:8,:]
        u_y_coding_predict = u_y_coding_predict.view(batch_size,-1,self.possion_num)
        sigma_x_coding_predict = cere_out[:,0,8:12,:]
        sigma_x_coding_predict = sigma_x_coding_predict.view(batch_size,-1,self.possion_num)
        sigma_y_coding_predict = cere_out[:,0,12:16,:]
        sigma_y_coding_predict = sigma_y_coding_predict.view(batch_size,-1,self.possion_num)

        m_x_coding_predict = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=50*(m_x+4)/4,dim=self.encoding_num)).float().to(self.device).float().to(self.device)
        m_y_coding_predict = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=50*(m_y+4)/4,dim=self.encoding_num)).float().to(self.device).float().to(self.device)

        sum_x_predict = self.network_x.forward(m_x_coding_predict,u_x_coding_predict,sigma_x_coding_predict,f_x_coding,time_window=self.possion_num)
        sum_y_predict = self.network_y.forward(m_y_coding_predict,u_y_coding_predict,sigma_y_coding_predict,f_y_coding,time_window=self.possion_num)
   
        sum_x_predict = sum_x_predict
        sum_mx = sum_x_predict[:,0:4]
        sum_ux = sum_x_predict[:,4:8]
        sum_y_predict = sum_y_predict
        sum_my = sum_y_predict[:,0:4]
        sum_uy = sum_y_predict[:,4:8]

        self.m_x = torch.sum(sum_mx)/100-2 +1
        self.m_y = torch.sum(sum_my)/100-2 +1

        final_output = torch.zeros([batch_size,3]).to(self.device)
        

        # the reverse is right?
        final_output[:,0] = -self.m_x
        final_output[:,1] = -self.m_y
        final_output[:,2] = self.m_dot_z_true

        #return final_output, out_ux_true, out_uy_true,self.snu_output_np[:,2], self.snu_output_np[:,3]
        return final_output

    def get_action(self,state):
        self.reservoir_network.reset()
        a_avg = self.forward(state)
        self.reservoir_network.reset()
        a_std = self.a_std_log.exp()
        action = a_avg + torch.randn_like(a_avg) * a_std
        return action
    
    def compute_logprob(self,state, action):
        a_avg = self.forward(state)
        action_std_log = self.a_std_log
        a_std = action_std_log.exp()
        delta = ((a_avg - action).mul(1/a_std)).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        logprob = -(action_std_log + self.sqrt_2pi_log + delta)
        return logprob.sum(dim=1)
    
    def get_a_std(self):
        action_std_log = self.a_std_log
        return action_std_log.exp()

    def save(self,path):
        torch.save(self.snu,f"{path}/m_snu_model_rl.pkl")
        torch.save(self.network_x,f"{path}/network_x_rl.pkl")
        torch.save(self.network_y,f"{path}/network_y_rl.pkl")
        torch.save(self.network,f"{path}/network_rl.pkl")
        sys.stderr.write("saved model successfully\n")
    
    def load(self,path):
        sys.stderr.write( "loading model from " + path + "\n")
        self.reservoir_network = torch.load(path+"/my_lsm_model_rl.pkl",encoding='unicode_escape')
        self.snu = torch.load(path+"my_snu_model_rl.pkl")
        sys.stderr.write("load lsm_snu model successfully\n")

        self.network_x = torch.load(path+"network_x_rl.pkl")
        self.network_y = torch.load(path+"network_x_rl.pkl")
        sys.stderr.write("load cere model successfully\n")

        self.network = torch.load(path+"network_rl.pkl")
        sys.stderr.write("load pre model successfully\n")

    def reset(self):
        #init matrix
        self.x_sum = 0
        self.train_in_spikes = np.zeros((self.ch, self.steps))
        self.snu_output_np = np.zeros([self.N_step,self.bias])

        self.possion_rate_coding = np.zeros([self.n_in,self.possion_num_snu])

        self.m_x = torch.tensor([0]).float().to(self.device)
        self.m_y = torch.tensor([0]).float().to(self.device)
        self.m_dot_z_true = torch.tensor([0]).float().to(self.device)
        self.belta_true = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=0.5*400/4/2,dim=4)).float().to(self.device)
        self.opts_mx = 0
        self.opts_my = 0
        self.opts_mz = 0

class CriticAdv(nn.Module):
    def __init__(self, state_dim=6,mid_dim=128,action_dim=3,if_use_dn=True,gpu='0',):
        super().__init__()
        self.state_dim = state_dim
        self.mid_dim = mid_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
        if if_use_dn:
            nn_dense = DenseNet(mid_dim).to(self.device)
            inp_dim = nn_dense.inp_dim
            out_dim = nn_dense.out_dim

            self.net = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                        nn_dense,
                                        nn.Linear(out_dim, 1), ).to(self.device)
        else:
            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                        nn.Linear(mid_dim, 1), ).to(self.device)

        layer_norm(self.net[-1], std=0.5)  # output layer for Q value

    def forward(self,state):
        a = self.net(state.to(self.device))  # Q value
        return a

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, mid_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(mid_dim // 2, mid_dim // 2), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(mid_dim * 1, mid_dim * 1), nn.Hardswish())
        self.inp_dim = mid_dim // 2
        self.out_dim = mid_dim * 2

    def forward(self, x1):  # x1.shape == (-1, mid_dim // 2)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3  # x3.shape == (-1, mid_dim * 2)


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

