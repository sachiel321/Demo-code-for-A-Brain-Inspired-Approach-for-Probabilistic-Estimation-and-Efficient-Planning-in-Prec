# -*- coding: utf-8 -*-
"""
@Author: YYM
@Institute: CASIA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy as np
import math

from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

from scipy import signal
from network.ObjectClasses import Neuron, Spike
from network.ReservoirDefinitions import create_random_reservoir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1)

thresh = 0.5
lens = 0.5
decay = 0.2
if_bias = True

################SNU model###################
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

    # @staticmethod
    # def backward(ctx, grad_h):
    #     z = ctx.saved_tensors
    #     s = torch.sigmoid(z[0])
    #     d_input = (1 - s) * s * grad_h
    #     return d_input

act_fun = ActFun.apply

def mem_update(ops, x, mem, spike, lateral = None):
    mem = mem * decay * (1. - spike) + ops(x)
    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem)
    return mem, spike
############################################
###############LSM_SNU model################
class SNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_classes, possion_num=19):
        super(SNN, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias = if_bias)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes, bias = if_bias)
        #torch.nn.init.eye_(self.fc1.weight)
        #torch.nn.init.eye_(self.fc2.weight)

        #monitor
        self.monitor_input = torch.zeros(self.batch_size, self.input_size, possion_num).to(device)
        self.monitor_fc1 = torch.zeros(self.batch_size, self.hidden_size, possion_num).to(device)
        self.monitor_fc2 = torch.zeros(self.batch_size, self.num_classes, possion_num).to(device)

    def forward(self, input, task, time_window):
        self.fc1 = self.fc1.float()
        self.fc2 = self.fc2.float()

        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.hidden_size).to(device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.num_classes).to(device)

        for step in range(time_window):

            x = input
            
            sum_out = None

            for t in range(time_window):
                
                if task == 'LSM':
                #x_t = torch.from_numpy(x[:,t]).float()
                    x_t = from_dlpack(toDlpack(x[:,t])).float()
                elif task == 'STDP':
                    x_t = x[:,t]


                x_t = x_t.to(device)

                x_t = x_t.view(self.batch_size, -1)
                self.monitor_input[:,:,t] = x_t
                h1_mem, h1_spike = mem_update(self.fc1, x_t, h1_mem, h1_spike)
                #h1_sumspike += h1_spike
                self.monitor_fc1[:,:,t] = h1_spike
                h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
                #h2_sumspike += h2_spike
                self.monitor_fc2[:,:,t] = h2_spike
                sum_out = h2_spike if sum_out is None else sum_out + h2_spike

        #outputs = h2_sumspike / time_window
        return sum_out
    def stdp_step(self,reward, lr):

        r_stdp(self.monitor_fc1[0],self.monitor_fc2[0],self.fc2.weight, reward, lr=lr)
        r_stdp(self.monitor_input[0],self.monitor_fc1[0],self.fc1.weight, reward, lr=lr)


class LSMNetwork:
    def __init__(self, dims, frac_inhibitory, w_matrix, fanout, 
                simulation_steps, num_in_ch, tau=20, t_ref=10, 
                propagation_time=10, ignore_frac=0.0,each_step_reset=True):
        #simulation_steps : total number of simulation steps to simulate time T in steps of dt = T/dt
        self.reset = each_step_reset
        self.ignore_frac = ignore_frac
        self.propagation_time = propagation_time
        self.tau = tau
        self.t_ref = t_ref
        self.dims = dims
        self.n_nodes = dims[0]*dims[1]*dims[2]
        self.num_in_ch = num_in_ch
        if num_in_ch<=self.n_nodes:
            mapped_nodes = np.random.choice(self.n_nodes, size=num_in_ch, replace=False)
        else:
            mapped_nodes = np.random.choice(self.n_nodes, size=num_in_ch, replace=True)
        self.mapped_nodes = mapped_nodes
        self.frac_inibitory = frac_inhibitory
        self.w_matrix = w_matrix
        self.fanout = fanout
        adj_mat, all_connections, all_weights = create_random_reservoir(dims, frac_inhibitory, w_matrix, fanout)
        #self.adj_mat = adj_mat
        self.all_connections = all_connections
        self.all_weights = all_weights
        self.neuronList = [Neuron(i, all_connections[i], all_weights[i], fanout, tau, t_ref, propagation_time) for i in range(len(all_connections))]
        self.simulation_steps = simulation_steps
        self.current_time_step = 0
        self.action_buffer = []
        for i in range(simulation_steps):
            self.action_buffer.append([])
        return
    
    def add_input(self, input_spike_train):
        #input_spike_train : num_channels x simulation_steps matrix of all channels of the input spike train
        # for i in range(len(self.neuronList)):
        #     self.neuronList[i] = Neuron(i, self.all_connections[i], self.all_weights[i], self.fanout, self.tau, self.t_ref, self.propagation_time)
        for t_step in range(input_spike_train.shape[1]):
            self.action_buffer[t_step] = []
            for ch in range(self.num_in_ch):
                if input_spike_train[ch,t_step] > 0:
                    self.action_buffer[t_step].append((input_spike_train[ch,t_step], self.mapped_nodes[ch]))
        return
    
    def simulate(self):
        rate_coding = np.zeros([self.n_nodes,self.simulation_steps])
        frac = self.ignore_frac
        for t_step in range(self.simulation_steps):
            #print(t_step)
            if len(self.action_buffer[t_step])>0:
                for action in self.action_buffer[t_step]:
                    spike_val = action[0]
                    target_node = action[1]
                    spike_produced = self.neuronList[int(target_node)].receive_spike(t_step, spike_val)
                    if spike_produced != None:
                        if t_step > frac*self.simulation_steps:
                            rate_coding[target_node][t_step] += 1
                        receiver_nodes = spike_produced.receiver_nodes
                        spike_values = spike_produced.spike_values
                        receive_times = spike_produced.receive_times
                        for node in range(len(receiver_nodes)):
                            if(receive_times[node]<self.simulation_steps):
                                self.action_buffer[int(receive_times[node])].append((int(spike_values[node]), receiver_nodes[node]))
        #if self.reset:
        #重置
        for i in range(len(self.neuronList)):
            self.neuronList[i].reset_spike()
        for step in range(self.simulation_steps):
            self.action_buffer[t_step] = []
        return rate_coding
############################################
##############cerebellar model##############
class cere_model(nn.Module):
    def __init__(self,batch_size,num_in_MF,
                    num_out_MF,num_out_GC,num_out_PC,num_out_DCN):
        super(cere_model,self).__init__()
        self.batch_size = batch_size
        self.num_out_MF = num_out_MF
        self.num_out_GC = num_out_GC
        self.num_out_PC = num_out_PC
        self.num_out_DCN = num_out_DCN
        #MF:
        self.m_MF = nn.Linear(num_in_MF*5, num_out_MF, bias = if_bias)
        self.u_MF = nn.Linear(num_in_MF*6, num_out_MF, bias = if_bias)
        self.sigma_MF = nn.Linear(num_in_MF*6, num_out_MF, bias = if_bias)
        self.f_MF = nn.Linear(num_in_MF*6, num_out_MF, bias = if_bias)

        #GC:
        num_in_GC=num_out_MF
        self.m_GC = nn.Linear(num_in_GC, num_out_GC, bias = if_bias)
        self.u_GC = nn.Linear(num_in_GC, num_out_GC, bias = if_bias)
        self.sigma_GC = nn.Linear(num_in_GC, num_out_GC, bias = if_bias)
        self.f_GC = nn.Linear(num_in_GC, num_out_GC, bias = if_bias)

        #PC:
        num_in_PC=4*num_out_GC
        #PC E
        self.m_PCE = nn.Linear(num_in_PC, num_out_PC, bias = if_bias)

        #PC I
        self.m_PCI = nn.Linear(num_in_PC, num_out_PC, bias = if_bias)

        #DCN:
        num_in_DCN=2*num_out_PC
        self.m_DCN = nn.Linear(num_in_DCN, num_out_DCN*11, bias = if_bias)
        #self.u_DCN = nn.Linear(num_in_DCN, num_out_DCN*4, bias = if_bias)

        #STDP prams
        
        #monitor
        self.monitor_MF_m = torch.zeros(self.batch_size, self.num_out_MF, 19).to(device)
        self.monitor_MF_u = torch.zeros(self.batch_size, self.num_out_MF, 19).to(device)
        self.monitor_MF_sigma = torch.zeros(self.batch_size, self.num_out_MF, 19).to(device)
        self.monitor_MF_f = torch.zeros(self.batch_size, self.num_out_MF, 19).to(device)

        self.monitor_GC_m = torch.zeros(self.batch_size, self.num_out_GC, 19).to(device)
        self.monitor_GC_u = torch.zeros(self.batch_size, self.num_out_GC, 19).to(device)
        self.monitor_GC_sigma = torch.zeros(self.batch_size, self.num_out_GC, 19).to(device)
        self.monitor_GC_f = torch.zeros(self.batch_size, self.num_out_GC, 19).to(device)

        self.monitor_PCE_m = torch.zeros(self.batch_size, self.num_out_PC, 19).to(device)
        self.monitor_PCI_m = torch.zeros(self.batch_size, self.num_out_PC, 19).to(device)

        self.monitor_DCN_m = torch.zeros(self.batch_size, self.num_out_DCN*11, 19).to(device)



    def forward(self,m,u,sigma,f, time_window):
        
        m_MF_mem = m_MF_spike = u_MF_mem = u_MF_spike = sigma_MF_mem = sigma_MF_spike = f_MF_mem = f_MF_spike = torch.zeros(self.batch_size, self.num_out_MF).to(device)
        m_GC_mem = m_GC_spike = u_GC_mem = u_GC_spike = sigma_GC_mem = sigma_GC_spike = f_GC_mem = f_GC_spike = torch.zeros(self.batch_size, self.num_out_GC).to(device)
        m_PCE_mem = m_PCE_spike= m_PCI_mem = m_PCI_spike = torch.zeros(self.batch_size, self.num_out_PC).to(device)
        m_DCN_mem = m_DCN_spike = torch.zeros(self.batch_size, self.num_out_DCN*11).to(device)

        
        if type(m) is np.ndarray:
                m = torch.from_numpy(m).float()
                u = torch.from_numpy(u).float()
                sigma = torch.from_numpy(sigma).float()
                f = torch.from_numpy(f).float()
        else:
            m = m.float()
            u = u.float()
            sigma = sigma.float()
            f = f.float()
        if torch.cuda.is_available():
            m = m.cuda()
            u = u.cuda()
            sigma = sigma.cuda()
            f = f.cuda()

        for step in range(time_window):
            sum_out_m = sum_out_u = None
            #x = x.float()
            #x = x.view(self.batch_size, -1)  
            #x_t = torch.from_numpy(x[:,t]).float().cuda()
            for t in range(time_window):
                m_t = m[:,t]
                m_t = m_t.view(self.batch_size, -1)
                u_t = u[:,t]
                u_t = u_t.view(self.batch_size, -1)
                sigma_t = sigma[:,t]
                sigma_t = sigma_t.view(self.batch_size, -1)
                f_t = f[:,t]
                f_t = f_t.view(self.batch_size, -1)
                #MF:
                m_MF_mem, m_MF_spike = mem_update(self.m_MF, m_t, m_MF_mem, m_MF_spike)
                u_MF_mem, u_MF_spike = mem_update(self.u_MF, u_t, u_MF_mem, u_MF_spike)
                sigma_MF_mem, sigma_MF_spike = mem_update(self.sigma_MF, sigma_t, sigma_MF_mem, sigma_MF_spike)
                f_MF_mem, f_MF_spike = mem_update(self.f_MF, f_t, f_MF_mem, f_MF_spike)

                self.monitor_MF_m[:,:,t] = m_MF_spike
                self.monitor_MF_u[:,:,t] = u_MF_spike
                self.monitor_MF_sigma[:,:,t] = sigma_MF_spike
                self.monitor_MF_f[:,:,t] = f_MF_spike

                #GC:
                m_GC_mem, m_GC_spike = mem_update(self.m_GC, m_MF_spike, m_GC_mem, m_GC_spike)
                u_GC_mem, u_GC_spike = mem_update(self.u_GC, u_MF_spike, u_GC_mem, u_GC_spike)
                sigma_GC_mem, sigma_GC_spike = mem_update(self.sigma_GC, sigma_MF_spike, sigma_GC_mem, sigma_GC_spike)
                f_GC_mem, f_GC_spike = mem_update(self.f_GC, f_MF_spike, f_GC_mem, f_GC_spike)

                self.monitor_GC_m[:,:,t] = m_GC_spike
                self.monitor_GC_u[:,:,t] = u_GC_spike
                self.monitor_GC_sigma[:,:,t] = sigma_GC_spike
                self.monitor_GC_f[:,:,t] = f_GC_spike

                #PC:
                #combine tensor
                PF_in = torch.cat((m_GC_spike,u_GC_spike,sigma_GC_spike,f_GC_spike),0)
                PF_in = PF_in.view(self.batch_size, -1)
                MF_in = torch.cat((m_MF_spike,u_MF_spike,sigma_MF_spike,f_MF_spike),0)
                MF_in = MF_in.view(self.batch_size, -1)


                m_PCE_mem, m_PCE_spike = mem_update(self.m_PCE, PF_in, m_PCE_mem, m_PCE_spike)
                m_PCI_mem, m_PCI_spike = mem_update(self.m_PCI, PF_in, m_PCI_mem, m_PCI_spike)

                self.monitor_PCE_m[:,:,t] = m_PCE_spike
                self.monitor_PCI_m[:,:,t] = m_PCI_spike

                #DCN:
                baseline_MF = torch.mean(MF_in) + 0.3
                #baseline_MF = 0
                m_DCN_in = torch.cat((m_PCE_spike,m_PCI_spike),0)
                m_DCN_in = m_DCN_in.view(self.batch_size, -1)
                m_DCN_in = m_DCN_in + baseline_MF

                m_DCN_mem, m_DCN_spike = mem_update(self.m_DCN, m_DCN_in, m_DCN_mem, m_DCN_spike)

                self.monitor_DCN_m[:,:,t] = m_DCN_spike

                sum_out_m = m_DCN_spike if sum_out_m is None else sum_out_m + m_DCN_spike

        return sum_out_m

    def stdp_step(self, reward, lr):

        r_stdp(self.monitor_PCE_m[0],self.monitor_DCN_m[0],self.m_DCN.weight[:,0:self.num_out_PC], reward,lr=lr)
        r_stdp(self.monitor_PCI_m[0],self.monitor_DCN_m[0],self.m_DCN.weight[:,self.num_out_PC:2 * self.num_out_PC], reward,lr=lr)

        r_stdp(self.monitor_GC_m[0],self.monitor_PCE_m[0],self.m_PCE.weight[:,0:self.num_out_GC], reward,lr=lr)
        r_stdp(self.monitor_GC_u[0],self.monitor_PCE_m[0],self.m_PCE.weight[:,self.num_out_GC:2*self.num_out_GC], reward,lr=lr)
        r_stdp(self.monitor_GC_sigma[0],self.monitor_PCE_m[0],self.m_PCE.weight[:,2*self.num_out_GC:3*self.num_out_GC], reward,lr=lr)
        r_stdp(self.monitor_GC_f[0],self.monitor_PCE_m[0],self.m_PCE.weight[:,3*self.num_out_GC:4*self.num_out_GC], reward,lr=lr)
        r_stdp(self.monitor_GC_m[0],self.monitor_PCI_m[0],self.m_PCI.weight[:,0:self.num_out_GC], reward,lr=lr)
        r_stdp(self.monitor_GC_u[0],self.monitor_PCI_m[0],self.m_PCI.weight[:,self.num_out_GC:2*self.num_out_GC], reward,lr=lr)
        r_stdp(self.monitor_GC_sigma[0],self.monitor_PCI_m[0],self.m_PCI.weight[:,2*self.num_out_GC:3*self.num_out_GC], reward,lr=lr)
        r_stdp(self.monitor_GC_f[0],self.monitor_PCI_m[0],self.m_PCI.weight[:,3*self.num_out_GC:4*self.num_out_GC], reward,lr=lr)




############################################
##############prefrontal model##############

class prefrontal_model(nn.Module):
    def __init__(self, batch_size, num_hidden1, num_hidden2, num_hidden3,N_step):
        super(prefrontal_model, self).__init__()
        self.batch_size = batch_size
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_hidden3 = num_hidden3
        self.inputnum1 = 4*6*N_step
        self.inputnum2 = 3*6
        self.inputnum3 = 7*6
        self.output1 = 1*6
        self.output2 = 1*6
        self.output3 = 2*8
        #layer
        self.fc11 = nn.Linear(self.inputnum1, self.num_hidden1, bias = if_bias)
        self.fc12 = nn.Linear(self.num_hidden1, self.output1, bias = if_bias)
        self.fc21 = nn.Linear(self.inputnum2, self.num_hidden2, bias = if_bias)
        self.fc22 = nn.Linear(self.num_hidden2, self.output2, bias = if_bias)
        self.fc31 = nn.Linear(self.inputnum3, self.num_hidden3, bias = if_bias)
        self.fc31_5 = nn.Linear(self.num_hidden3, self.num_hidden3, bias = if_bias)
        self.fc32 = nn.Linear(self.num_hidden3, self.output3, bias = if_bias)

        self.layer3_in = torch.zeros(1,self.inputnum3)

        #monitor
        self.monitor_h11 = torch.zeros(self.batch_size, self.num_hidden1, 19).to(device)
        self.monitor_h12 = torch.zeros(self.batch_size, self.output1, 19).to(device)
        self.monitor_h21 = torch.zeros(self.batch_size, self.num_hidden2, 19).to(device)
        self.monitor_h22 = torch.zeros(self.batch_size, self.output2, 19).to(device)
        self.monitor_layer3_in = torch.zeros(self.batch_size, self.inputnum3, 19).to(device)
        self.monitor_h31 = torch.zeros(self.batch_size, self.num_hidden3, 19).to(device)
        self.monitor_h31_5 = torch.zeros(self.batch_size, self.num_hidden3, 19).to(device)
        self.monitor_h32 = torch.zeros(self.batch_size, self.output3, 19).to(device)
    
    def forward(self,input11,input12,input2,input3,input4, time_window):
        # input1: (f(x,t), f(y,t))
        # input2: (u(x,t+n), sigma(x,t+n), u(y,t+n), sigma(y,t+n))
        # input3: (delta(z), m(z,t-1), m_dot(z,t-1))
        # input4: belta(t-1)
        h11_mem = h11_spike = h11_sumspike = torch.zeros(self.batch_size, self.num_hidden1).to(device)
        h12_mem = h12_spike = h12_sumspike = torch.zeros(self.batch_size, self.output1).to(device)
        h21_mem = h21_spike = h21_sumspike = torch.zeros(self.batch_size, self.num_hidden2).to(device)
        h22_mem = h22_spike = h22_sumspike = torch.zeros(self.batch_size, self.output2).to(device)
        h31_mem = h31_spike = h31_sumspike = torch.zeros(self.batch_size, self.num_hidden3).to(device)
        h31_5_mem = h31_5_spike = h31_5_sumspike = torch.zeros(self.batch_size, self.num_hidden3).to(device)
        h32_mem = h32_spike = h32_sumspike = torch.zeros(self.batch_size, self.output3).to(device)
    
        if type(input11) is np.ndarray:
                input11 = torch.from_numpy(input11).float()
                input12 = torch.from_numpy(input12).float()
                input2 = torch.from_numpy(input2).float()
                input3 = torch.from_numpy(input3).float()
                input4 = torch.from_numpy(input4).float()
        else:
            input11 = input11.float()
            input12 = input12.float()
            input2 = input2.float()
            input3 = input3.float()
            input4 = input4.float()

        input11 = input11.to(device)
        input12 = input12.to(device)
        input2 = input2.to(device)
        input3 = input3.to(device)
        input4 = input4.to(device)

        for step in range(time_window):
            sum_out1 = None
            sum_out2 = None
            sum_out3 = None
            #x = x.view(self.batch_size, -1)

            for t in range(time_window):
                #get signal
                #x_t = torch.from_numpy(x[:,t]).float().cuda()
                input11_t = input11[:,t].float()
                input11_t = input11_t.view(self.batch_size, -1)
                input12_t = input12[:,t].float()
                input12_t = input12_t.view(self.batch_size, -1)
                input2_t = input2[:,t].float()
                input2_t = input2_t.view(self.batch_size, -1)
                input3_t = input3[:,t].float()
                input3_t = input3_t.view(self.batch_size, -1)
                input4_t = input4[:,t].float()
                input4_t = input4_t.view(self.batch_size, -1)

                #layer forward
                h11_mem, h11_spike = mem_update(self.fc11, input2_t, h11_mem, h11_spike)
                #h11_sumspike += h11_spike
                self.monitor_h11[:,:,t] = h11_spike
                h12_mem, h12_spike = mem_update(self.fc12, h11_spike, h12_mem, h12_spike)
                #h12_sumspike += h12_spike
                self.monitor_h12[:,:,t] = h12_spike

                h21_mem, h21_spike = mem_update(self.fc21, input3_t, h21_mem, h21_spike)
                #h21_sumspike += h21_spike
                self.monitor_h21[:,:,t] = h21_spike
                h22_mem, h22_spike = mem_update(self.fc22, h21_spike, h22_mem, h22_spike)
                #h22_sumspike += h22_spike
                self.monitor_h22[:,:,t] = h22_spike

                self.layer3_in = torch.cat([input11_t,input11_t,h12_spike,h22_spike,input3[12:18,t].view(self.batch_size, -1),input4_t],1)
                self.monitor_layer3_in[:,:,t] = self.layer3_in
                h31_mem, h31_spike = mem_update(self.fc31, self.layer3_in, h31_mem, h31_spike)
                #h31_sumspike += h31_spike
                self.monitor_h31[:,:,t] = h31_spike
                h31_5_mem, h31_5_spike = mem_update(self.fc31_5, h31_spike, h31_5_mem, h31_5_spike)
                #h32_sumspike += h32_spike
                self.monitor_h31_5[:,:,t] = h31_5_spike

                h32_mem, h32_spike = mem_update(self.fc32, h31_5_spike, h32_mem, h32_spike)
                #h32_sumspike += h32_spike
                self.monitor_h32[:,:,t] = h32_spike


                sum_out1 = h12_spike if sum_out1 is None else sum_out1 + h12_spike
                sum_out2 = h22_spike if sum_out2 is None else sum_out2 + h22_spike
                sum_out3 = h32_spike if sum_out3 is None else sum_out3 + h32_spike

        #outputs = h2_sumspike / time_window
        return sum_out1, sum_out2, sum_out3
    
    def stdp_step(self, reward,lr):

        r_stdp(self.monitor_h31[0],self.monitor_h32[0],self.fc32.weight, reward,lr=lr)
        r_stdp(self.monitor_layer3_in[0],self.monitor_h31[0],self.fc31.weight, reward,lr=lr)



############################################


def r_stdp(pre_spike, post_spike, weight, reward, lr=0.03):
    #pre_spike: (pre_neuron_num, timescale)
    #post_spike: (post_neuron_num, timescale)
    #layer.weight: (post_neuron_num, pre_neuron_num)

    A_positve = 1
    A_negative = -1
    tao_positive = 20
    tao_negative = 20
    tao_z = 25

    if pre_spike.size()[1] != post_spike.size()[1] or pre_spike.size()[0] != weight.data.size()[1] or post_spike.size()[0] != weight.data.size()[0]:
        print('matrix dimention error')
        return False

    pre_size = pre_spike.size()[0]
    post_size = post_spike.size()[0]
    timescale = pre_spike.size()[1]

    P_positive = torch.zeros_like(pre_spike[:,0]).view(1,-1)
    P_negative = torch.zeros_like(post_spike[:,0]).view(1,-1)

    z = 0
    dt = 1
    
    for t in range(0,timescale):
        temp_pre = pre_spike[:,t].view(1,-1)
        temp_post = post_spike[:,t].view(1,-1)
        P_positive = -P_positive * t * math.exp(-dt/tao_positive) + A_positve * temp_pre
        P_negative = -P_negative * t * math.exp(-dt/tao_negative) + A_negative * temp_post

        temp_post_transpose = torch.t(temp_post)
        P_negative_transpose = torch.t(P_negative)
        kesai =  torch.mm(temp_post_transpose, P_positive) + torch.mm(P_negative_transpose, temp_pre)
        z = z * math.exp(-dt/tao_z) + kesai
        weight.data = weight.data + lr * reward * z
        weight.data = torch.clamp(weight.data, -100,100)
    
    return True

