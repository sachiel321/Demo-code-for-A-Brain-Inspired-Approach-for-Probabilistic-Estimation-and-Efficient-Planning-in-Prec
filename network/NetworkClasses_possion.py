# -*- coding: utf-8 -*-
"""
@Author: YYM
@Institute: CASIA
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.jit as jit
import math

from scipy import signal
from network.ObjectClasses import Neuron, Spike
from network.ReservoirDefinitions import create_random_reservoir

thresh = 0.5
dampening_factor = 0.3
lens = 0.5
decay = 0.2
if_bias = True

def to_device(input):
    return input.cuda()

################SNU model###################
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, a=thresh):
        ctx.save_for_backward(input)
        return input.gt(a).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = torch.max((1 - torch.abs(input/thresh)), to_device(torch.tensor(0)).float()) * dampening_factor
        return grad_input * temp.float(), None

    # @staticmethod
    # def backward(ctx, grad_h):
    #     z = ctx.saved_tensors
    #     s = torch.sigmoid(z[0])
    #     d_input = (1 - s) * s * grad_h
    #     return d_input

act_fun = ActFun.apply

def mem_update(ops, x, v_mem, spike, lateral = None):
    v_mem = v_mem * decay * (1. - spike) + ops(x)
    if lateral:
        v_mem += lateral(spike)
    spike = act_fun(v_mem)
    return v_mem, spike
############################################
###############LSM_SNU model################
class SNN(nn.Module):
    def __init__(self, batch_size, input_size, num_classes, encoding_num=4, possion_num=50, gpu='0'):
        super(SNN, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes*encoding_num #every class coded by 4 neurons
        self.possion_num = possion_num
        self.fc1 = nn.Linear(self.input_size, self.num_classes, bias = if_bias)
        self.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")

    def forward(self, input, task, time_window):
        self.fc1 = self.fc1.float()
        batch_size = input.shape[0]
        #monitor
        self.monitor_input = torch.zeros(batch_size, self.input_size, self.possion_num).cuda()
        self.monitor_fc1 = torch.zeros(batch_size, self.num_classes, self.possion_num).cuda()
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, self.num_classes).cuda()

        for step in range(time_window):
            
            x = input
            
            sum_out = None

            for t in range(time_window):
                if task == "LSM":
                    x_t = x[:,t].float().cuda()
                elif task == 'STDP':
                    x_t = x[:,t].cuda()

                x_t = x_t.view(batch_size, -1)
                h1_mem, h1_spike = mem_update(self.fc1, x_t, h1_mem, h1_spike)
                #h1_sumspike += h1_spike
                
                with torch.no_grad():
                    self.monitor_fc1[:,:,t] = h1_spike.detach()
                sum_out = h1_spike if sum_out is None else sum_out + h1_spike

        #outputs = h2_sumspike / time_window
        return sum_out

    def stdp_step(self,reward, lr):
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
            mapped_nodes = torch.from_numpy(np.random.choice(self.n_nodes, size=num_in_ch, replace=False))
        else:
            mapped_nodes = torch.from_numpy(np.random.choice(self.n_nodes, size=num_in_ch, replace=True))
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
        #TODO: Replace list in a more efficient way
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
                    #TODO: Replace list in a more efficient way
                    self.action_buffer[t_step].append((input_spike_train[ch,t_step], self.mapped_nodes[ch]))
        return
    
    def simulate(self):
        rate_coding = torch.zeros([self.n_nodes,self.simulation_steps])
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
                                #TODO: Replace list in a more efficient way
                                self.action_buffer[int(receive_times[node])].append((int(spike_values[node]), receiver_nodes[node]))
        #if self.reset:
        #reset
        for i in range(len(self.neuronList)):
            self.neuronList[i].reset_spike()
        for step in range(self.simulation_steps):
            self.action_buffer[t_step] = []
        return rate_coding
############################################
#################e-prop#####################

class EPropBase(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                input, 
                spike, 
                v_hidden,
                a, 
                weight_hh, 
                weight_ih, 
                e_trace_hh,
                ev_w_hh_x,
                ea_w_hh_x,
                e_trace_ih,
                ev_w_ih_x,
                ea_w_ih_x,
                gamma_pd,
                thresh,
                alpha,
                beta,
                rho):   
        # TODO:Not on the same device
        # Remove the main diagonal element
        temp_w_hh = weight_hh - to_device(torch.eye(weight_hh.size(0))) * weight_hh
        v_hidden_t = alpha * v_hidden + torch.mm(spike, temp_w_hh) + torch.mm(input, weight_ih) - v_hidden.gt(a).float() * a

        spike_t = act_fun(v_hidden_t,a)
        a = rho * a + spike_t       
        A = thresh + beta * a

        psi = 1/thresh * gamma_pd * torch.max(to_device(torch.tensor(0)).float(), (1-torch.abs(1/thresh*(v_hidden_t-A))))

        spike_for_hh = spike.resize(input.size(0),weight_ih.size(1),1).repeat(1,1,weight_ih.size(1))
        #TODO: Compare the current spike or the previous spike
        ev_w_hh_x = alpha * ev_w_hh_x + spike_for_hh
        e_trace_hh = psi[:,None,:] * (ev_w_hh_x - beta * ea_w_hh_x)
        ea_w_hh_x = psi[:,None,:] * ev_w_hh_x + (rho - psi[:,None,:] * beta) * ea_w_hh_x

        spike_for_ih = input.resize(input.size(0),weight_ih.size(0),1).repeat(1,1,weight_ih.size(1))

        ev_w_ih_x = alpha * ev_w_ih_x + spike_for_ih
        e_trace_ih = psi[:,None,:] * (ev_w_ih_x - beta * ea_w_ih_x)

        ea_w_ih_x = psi[:,None,:] * ev_w_ih_x + (rho - psi[:,None,:] * beta) * ea_w_ih_x

        ctx.save_for_backward(e_trace_hh, e_trace_ih)

        return spike_t, v_hidden_t, A, e_trace_hh,ev_w_hh_x,ea_w_hh_x,e_trace_ih,ev_w_ih_x,ea_w_ih_x

    @staticmethod
    def backward(ctx, grad_spike_t, grad_hidden_t, grad_A, grad_e_trace_hh,grad_ev_w_hh_x,grad_ea_w_hh_x,grad_e_trace_ih,grad_ev_w_ih_x,grad_ea_w_ih_x):
        #TODO: Rewrite backward
        e_trace_hh, e_trace_ih,  = ctx.saved_variables
        grad_weight_ih = e_trace_ih * grad_spike_t.reshape(grad_spike_t.size(0),1,grad_spike_t.size(1)).repeat(1,e_trace_ih.shape[1],1)
        grad_weight_hh = e_trace_hh * grad_spike_t.reshape(grad_spike_t.size(0),1,grad_spike_t.size(1)).repeat(1,e_trace_hh.shape[1],1)

        #input, spike, v_hidden, a, weight_hh, weight_ih, e_trace_hh, ev_w_hh_x, ea_w_hh_x, e_trace_ih, ev_w_ih_x, ea_w_ih_x, gamma_pd, thresh, alpha, beta, rho
        return None, None, None, None, grad_weight_hh, grad_weight_ih, None,None,None,None,None,None, None, None, None, None, None

eprop = EPropBase.apply

###################Recurrent################

class RSNU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False):
        super(RSNU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Create for weight matrices, one for each gate + recurrent connections
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None

        self.initialize_parameters(self.weight_ih, self.bias_ih)
        self.initialize_parameters(self.weight_hh, self.bias_hh)

    def initialize_parameters(self, weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

class EPropRSNU(RSNU):
    def __init__(self, input_size, hidden_size, bias=True, thresh=1, beta=0.1, dt=1,tau=10, tau_e=70,gpu='0'):
        super(EPropRSNU, self).__init__(input_size, hidden_size, bias)
        self.thresh = thresh
        self.beta = beta
        self.dt = dt
        self.tau = torch.tensor(tau).float()
        self.tau_e = torch.tensor(tau_e).float()
        self.alpha = torch.exp(-self.dt / self.tau)
        self.rho = torch.exp(-self.dt / self.tau_e)
        self.a = None
        self.batch_size = None
        self.v_hidden = None
        self.eligibility_vectors = []
        self.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
    def forward(self, input, initial_v_h):
        '''
        input (batch x, input_size, seq_len x)
        initial_hidden (batch x, hidden_size)
        initial_state (batch x, hidden_size)
        
        cut input into slices follow dim 0(time)
        '''
        input = input.to(self.device)

        inputs = input.unbind(2)

        input_size = input.size(1)
        self.hidden_size = initial_v_h.size(1)
        self.batch_size = input.size(0)
        batch_size = input.size(0)
        if self.v_hidden == None:
            self.v_hidden = initial_v_h.to(self.device)

        spike = torch.zeros_like(self.v_hidden)

        if self.a == None:
            self.a = torch.zeros_like(self.v_hidden)

        if len(self.eligibility_vectors) == 0:
            ev_w_hh_x = torch.zeros(batch_size, self.hidden_size, self.hidden_size, requires_grad=False).to(self.device)
            ea_w_hh_x = torch.zeros_like(ev_w_hh_x)
            e_trace_hh = torch.zeros_like(ev_w_hh_x)

            ev_w_ih_x = torch.zeros(batch_size, self.input_size, self.hidden_size, requires_grad=False).to(self.device)
            ea_w_ih_x = torch.zeros_like(ev_w_ih_x)
            e_trace_ih = torch.zeros_like(ev_w_ih_x)
            self.eligibility_vectors = [e_trace_hh, ev_w_hh_x, ea_w_hh_x, e_trace_ih, ev_w_ih_x, ea_w_ih_x]


        outputs = []
        gamma_pd=0.3
        for i in range(len(inputs)):
            spike, self.v_hidden, self.a, self.eligibility_vectors[0], self.eligibility_vectors[1], self.eligibility_vectors[2], self.eligibility_vectors[3], self.eligibility_vectors[4], self.eligibility_vectors[5] = eprop(inputs[i], spike, self.v_hidden, self.a, self.weight_hh, self.weight_ih, self.eligibility_vectors[0], self.eligibility_vectors[1], self.eligibility_vectors[2], self.eligibility_vectors[3], self.eligibility_vectors[4], self.eligibility_vectors[5], gamma_pd, self.thresh, self.alpha, self.beta, self.rho)
            outputs += [spike]

        return torch.stack(outputs)

    def reset(self):
        self.a = None
        self.v_hidden = None
        self.eligibility_vectors = []



##############cerebellar model##############

def gen_mask(row, col, percent=0.5, num_ones=None):
    if num_ones is None:
        # Total number being masked is 0.5 by default.
        num_ones = int(row * percent)
    
    mask = np.zeros([row,col])
    for i in range(col):
        temp_mask = np.hstack([
    	np.zeros(num_ones),
        np.ones(row - num_ones)])
        np.random.shuffle(temp_mask)
        mask[:,i] = temp_mask
    return mask.reshape(row, col)

class SparseLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask

        # if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class SparseLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, mask=None):
        """
        Argumens
        ------------------
        mask [numpy.array]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(SparseLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute.
        self.weight = nn.Parameter(torch.Tensor(
            self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(
            	torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Initialize the above parameters (weight & bias).
        self.init_params()

        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float).t()
            self.mask = nn.Parameter(mask, requires_grad=False)
            # print('\n[!] CustomizedLinear: \n', self.weight.data.t())
        else:
            self.register_parameter('mask', None)

    def init_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return SparseLinearFunction.apply(
        	input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}, mask={}'.format(
            self.input_features, self.output_features,
            self.bias is not None, self.mask is not None)


class cere_SNN(nn.Module):
    def __init__(self,batch_size,num_in_MF,num_out_MF,num_out_GC,num_out_PC,num_out_DCN,possion_num=50):
        super(cere_SNN, self).__init__()
        self.batch_size = batch_size
        self.num_in_MF = num_in_MF*16
        self.num_out_MF = num_out_MF*4
        self.num_out_GC = num_out_GC*4
        self.num_out_PC = num_out_PC*4
        self.num_out_DCN = num_out_DCN*8
        self.possion_num = possion_num
        MF_GC_mask = gen_mask(self.num_out_MF, self.num_out_GC,num_ones=4)
        self.fc1 = nn.Linear(self.num_in_MF, self.num_out_MF, bias = if_bias)
        self.fc2 = SparseLinear(self.num_out_MF, self.num_out_GC, bias = if_bias,mask=MF_GC_mask)
        self.fc3 = nn.Linear(self.num_out_GC, self.num_out_PC, bias = if_bias)
        self.fc4 = nn.Linear(self.num_out_PC, self.num_out_DCN, bias = if_bias)

        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self,m,u,sigma,f, time_window):
        self.fc1 = self.fc1.float()
        # self.fc3.weight.data = torch.clamp(self.fc3.weight, -100, 0)
        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.num_out_MF).cuda()
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.num_out_GC).cuda()
        h3_mem = h3_spike = h3_sumspike = torch.zeros(self.batch_size, self.num_out_PC).cuda()
        h4_mem = h4_spike = h4_sumspike = torch.zeros(self.batch_size, self.num_out_DCN).cuda()
        
        input = torch.cat((m,u,sigma,f),1)
        x = input
        sum_out = None
        for t in range(time_window):
            x_t = x[:,:,t]
            x_t = x_t.view(self.batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x_t, h1_mem, h1_spike)
            #h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            #h1_sumspike += h1_spike
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            #h1_sumspike += h1_spike

            #DCN:
            baseline_MF = torch.mean(h1_spike)
            m_DCN_in = -h3_spike + torch.abs(baseline_MF)

            h4_mem, h4_spike = mem_update(self.fc4, m_DCN_in, h4_mem, h4_spike)
            #h1_sumspike += h1_spike
            
            sum_out = h4_spike if sum_out is None else sum_out + h4_spike

        return sum_out

class cere_model(nn.Module):
    def __init__(self,batch_size,num_in_MF,
                    num_out_MF,num_out_GC,num_out_PC,num_out_DCN,possion_num=50,gpu='0'):
        super(cere_model,self).__init__()
        self.batch_size = batch_size
        self.num_out_MF = num_out_MF
        self.num_out_GC = num_out_GC
        self.num_out_PC = num_out_PC
        self.num_out_DCN = num_out_DCN
        self.possion_num = possion_num
        self.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
        #MF: every class coded by 4 neurons
        self.m_MF = nn.Linear(num_in_MF*4, num_out_MF, bias = if_bias)
        self.u_MF = nn.Linear(num_in_MF*4, num_out_MF, bias = if_bias)
        self.sigma_MF = nn.Linear(num_in_MF*4, num_out_MF, bias = if_bias)
        self.f_MF = nn.Linear(num_in_MF*4, num_out_MF, bias = if_bias)

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

        #DCN: every class coded by 4 neurons
        num_in_DCN=num_out_PC
        self.m_DCN = nn.Linear(num_in_DCN, num_out_DCN*8, bias = if_bias)
        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # #STDP prams        
        # #monitor
        # self.monitor_MF_m = torch.zeros(self.batch_size, self.num_out_MF, self.possion_num).to(self.device)
        # self.monitor_MF_u = torch.zeros(self.batch_size, self.num_out_MF, self.possion_num).to(self.device)
        # self.monitor_MF_sigma = torch.zeros(self.batch_size, self.num_out_MF, self.possion_num).to(self.device)
        # self.monitor_MF_f = torch.zeros(self.batch_size, self.num_out_MF, self.possion_num).to(self.device)

        # self.monitor_GC_m = torch.zeros(self.batch_size, self.num_out_GC, self.possion_num).to(self.device)
        # self.monitor_GC_u = torch.zeros(self.batch_size, self.num_out_GC, self.possion_num).to(self.device)
        # self.monitor_GC_sigma = torch.zeros(self.batch_size, self.num_out_GC, self.possion_num).to(self.device)
        # self.monitor_GC_f = torch.zeros(self.batch_size, self.num_out_GC, self.possion_num).to(self.device)
        # self.monitor_PCE_m = torch.zeros(self.batch_size, self.num_out_PC, self.possion_num).to(self.device)
        # self.monitor_PCI_m = torch.zeros(self.batch_size, self.num_out_PC, self.possion_num).to(self.device)

        # self.monitor_DCN_m = torch.zeros(self.batch_size, self.num_out_DCN*8, self.possion_num).to(self.device)


    def forward(self,m,u,sigma,f, time_window):
        batch_size = m.shape[0]    
        self.monitor_DCN_m = torch.zeros(batch_size, self.num_out_DCN*8, self.possion_num).cuda()
        m_MF_mem = m_MF_spike = u_MF_mem = u_MF_spike = sigma_MF_mem = sigma_MF_spike = f_MF_mem = f_MF_spike = torch.zeros(batch_size, self.num_out_MF).cuda()
        m_GC_mem = m_GC_spike = u_GC_mem = u_GC_spike = sigma_GC_mem = sigma_GC_spike = f_GC_mem = f_GC_spike = torch.zeros(batch_size, self.num_out_GC).cuda()
        m_PCE_mem = m_PCE_spike = u_PCE_mem = u_PCE_spike = m_PCI_mem = m_PCI_spike = u_PCI_mem = u_PCI_spike =  torch.zeros(batch_size, self.num_out_PC).cuda()
        m_DCN_mem = m_DCN_spike = torch.zeros(batch_size, self.num_out_DCN*8).cuda()
        
        m = m.float()
        u = u.float()
        sigma = sigma.float()
        f = f.float()

        for step in range(time_window):
            sum_out_m = sum_out_u = None
            #x = x.float()
            #x = x.view(self.batch_size, -1)  
            #x_t = torch.from_numpy(x[:,t]).float().to(self.device)
            for t in range(time_window):
                m_t = m[:,:,t]
                m_t = m_t.view(batch_size, -1)
                u_t = u[:,:,t]
                u_t = u_t.view(batch_size, -1)
                sigma_t = sigma[:,:,t]
                sigma_t = sigma_t.view(batch_size, -1)
                f_t = f[:,:,t]
                f_t = f_t.view(batch_size, -1)
                #MF:
                m_MF_mem, m_MF_spike = mem_update(self.m_MF, m_t, m_MF_mem, m_MF_spike)
                u_MF_mem, u_MF_spike = mem_update(self.u_MF, u_t, u_MF_mem, u_MF_spike)
                sigma_MF_mem, sigma_MF_spike = mem_update(self.sigma_MF, sigma_t, sigma_MF_mem, sigma_MF_spike)
                f_MF_mem, f_MF_spike = mem_update(self.f_MF, f_t, f_MF_mem, f_MF_spike)
                
                #GC:
                m_GC_mem, m_GC_spike = mem_update(self.m_GC, m_MF_spike, m_GC_mem, m_GC_spike)
                u_GC_mem, u_GC_spike = mem_update(self.u_GC, u_MF_spike, u_GC_mem, u_GC_spike)
                sigma_GC_mem, sigma_GC_spike = mem_update(self.sigma_GC, sigma_MF_spike, sigma_GC_mem, sigma_GC_spike)
                f_GC_mem, f_GC_spike = mem_update(self.f_GC, f_MF_spike, f_GC_mem, f_GC_spike)
                
                #PC:
                #combine tensor
                PF_in = torch.cat((m_GC_spike,u_GC_spike,sigma_GC_spike,f_GC_spike),0)
                PF_in = PF_in.view(batch_size, -1)
                MF_in = torch.cat((m_MF_spike,u_MF_spike,sigma_MF_spike,f_MF_spike),0)
                MF_in = MF_in.view(batch_size, -1)                

                m_PCE_mem, m_PCE_spike = mem_update(self.m_PCE, PF_in, m_PCE_mem, m_PCE_spike)

                m_PCI_mem, m_PCI_spike = mem_update(self.m_PCI, PF_in, m_PCI_mem, m_PCI_spike)
                
                #DCN:
                baseline_MF = torch.mean(MF_in)
                m_DCN_in = m_PCE_spike + m_PCI_spike + baseline_MF

                m_DCN_mem, m_DCN_spike = mem_update(self.m_DCN, m_DCN_in, m_DCN_mem, m_DCN_spike)

                # with torch.no_grad():
                #     self.monitor_MF_m[:,:,t] = m_MF_spike.detach()
                #     self.monitor_MF_u[:,:,t] = u_MF_spike.detach()
                #     self.monitor_MF_sigma[:,:,t] = sigma_MF_spike.detach()
                #     self.monitor_MF_f[:,:,t] = f_MF_spike.detach()
                #     self.monitor_GC_m[:,:,t] = m_GC_spike.detach()
                #     self.monitor_GC_u[:,:,t] = u_GC_spike.detach()
                #     self.monitor_GC_sigma[:,:,t] = sigma_GC_spike.detach()
                #     self.monitor_GC_f[:,:,t] = f_GC_spike.detach()
                #     self.monitor_PCE_m[:,:,t] = m_PCE_spike.detach()
                #     self.monitor_PCI_m[:,:,t] = m_PCI_spike.detach()
                self.monitor_DCN_m[:,:,t] = m_DCN_spike

                sum_out_m = m_DCN_spike if sum_out_m is None else sum_out_m + m_DCN_spike

        return sum_out_m#, self.monitor_DCN_m
    
    # def stdp_step(self, reward, lr):

    #     r_stdp(self.monitor_PCE_m[0],self.monitor_DCN_m[0],self.m_DCN.weight[:,0:self.num_out_PC], reward,lr=lr)
    #     r_stdp(self.monitor_PCI_m[0],self.monitor_DCN_m[0],self.m_DCN.weight[:,self.num_out_PC:2 * self.num_out_PC], reward,lr=lr)

    #     r_stdp(self.monitor_GC_m[0],self.monitor_PCE_m[0],self.m_PCE.weight[:,0:self.num_out_GC], reward,lr=lr)
    #     r_stdp(self.monitor_GC_u[0],self.monitor_PCE_m[0],self.m_PCE.weight[:,self.num_out_GC:2*self.num_out_GC], reward,lr=lr)
    #     r_stdp(self.monitor_GC_sigma[0],self.monitor_PCE_m[0],self.m_PCE.weight[:,2*self.num_out_GC:3*self.num_out_GC], reward,lr=lr)
    #     r_stdp(self.monitor_GC_f[0],self.monitor_PCE_m[0],self.m_PCE.weight[:,3*self.num_out_GC:4*self.num_out_GC], reward,lr=lr)
    #     r_stdp(self.monitor_GC_m[0],self.monitor_PCI_m[0],self.m_PCI.weight[:,0:self.num_out_GC], reward,lr=lr)
    #     r_stdp(self.monitor_GC_u[0],self.monitor_PCI_m[0],self.m_PCI.weight[:,self.num_out_GC:2*self.num_out_GC], reward,lr=lr)
    #     r_stdp(self.monitor_GC_sigma[0],self.monitor_PCI_m[0],self.m_PCI.weight[:,2*self.num_out_GC:3*self.num_out_GC], reward,lr=lr)
    #     r_stdp(self.monitor_GC_f[0],self.monitor_PCI_m[0],self.m_PCI.weight[:,3*self.num_out_GC:4*self.num_out_GC], reward,lr=lr)

############################################
##############prefrontal model##############

class prefrontal_model_FF(nn.Module):
    def __init__(self, batch_size, num_hidden,N_step,possion_num=50,gpu='0'):
        super(prefrontal_model, self).__init__()
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.inputnum = 16*4
        self.output = 4
        self.possion_num = possion_num
        self.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
        #layer
        self.fc1 = nn.Linear(self.inputnum, self.num_hidden, bias = if_bias)
        self.fc2 = nn.Linear(self.num_hidden, 2*self.num_hidden, bias = if_bias)
        self.fc3 = nn.Linear(2*self.num_hidden, self.num_hidden, bias = if_bias)
        self.fc4 = nn.Linear(self.num_hidden, self.num_hidden, bias = if_bias)
        self.fc5 = nn.Linear(self.num_hidden, self.output, bias = if_bias)
        #monitor
        self.monitor_h1 = torch.zeros(self.batch_size, self.num_hidden, self.possion_num).to(self.device)
    
    def forward(self,input, time_window):
        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, 2*self.num_hidden).to(self.device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
        h4_mem = h4_spike = h4_sumspike = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
        h5_mem = h5_spike = h5_sumspike = torch.zeros(self.batch_size, self.output).to(self.device)

        for step in range(time_window):
            sum_out1 = None
            sum_out2 = None
            sum_out3 = None
            #x = x.view(self.batch_size, -1)

            for t in range(time_window):
                #get signal
                #x_t = torch.from_numpy(x[:,t]).float().cuda()
                input_t = input[:,t].float()
                input_t = input_t.view(self.batch_size, -1)

                #layer forward
                h1_mem, h1_spike = mem_update(self.fc1, input_t, h1_mem, h1_spike)
                #h1_sumspike += h1_spike
                
                h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
                #h2_sumspike += h2_spike
                
                h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
                #h3_sumspike += h3_spike

                h4_mem, h4_spike = mem_update(self.fc4, h3_spike, h4_mem, h4_spike)
                #h4_sumspike += h4_spike

                h5_mem, h5_spike = mem_update(self.fc5, h4_spike, h5_mem, h5_spike)
                #h5_sumspike += h5_spike

                sum_out = h5_spike if sum_out is None else sum_out + h5_spike

        #outputs = h2_sumspike / time_window
        return sum_out

class prefrontal_model(nn.Module):
    def __init__(self, batch_size, num_hidden1, num_hidden2, num_hidden3,N_step,possion_num=50,gpu='0'):
        super(prefrontal_model, self).__init__()
        self.batch_size = batch_size
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_hidden3 = num_hidden3
        self.inputnum = 16*4
        self.output = 4
        self.possion_num = possion_num
        self.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
        #layer
        self.fc1 = nn.Linear(self.inputnum, self.num_hidden1, bias = if_bias)
        self.fc2 = nn.Linear(self.num_hidden1, self.num_hidden2, bias = if_bias)
        self.fc3 = nn.Linear(self.num_hidden2, self.num_hidden2, bias = if_bias)
        self.fc4 = nn.Linear(self.num_hidden2, self.num_hidden3, bias = if_bias)
        self.fc5 = nn.Linear(self.num_hidden3, self.output, bias = if_bias)
   
    def forward(self,input, time_window):
        '''
        input1: (f(x,t), f(y,t))
        input2: (u(x,t+n), sigma(x,t+n), u(y,t+n), sigma(y,t+n))
        input3: (delta(z), m(z,t-1), m_dot(z,t-1),speed_limiter)
        input4: m_dot(z,t-1)
        '''
        batch_size = input.shape[0]
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, self.num_hidden1).to(self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, self.num_hidden2).to(self.device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, self.num_hidden2).to(self.device)
        h4_mem = h4_spike = h4_sumspike = torch.zeros(batch_size, self.num_hidden3).to(self.device)
        h5_mem = h5_spike = h5_sumspike = torch.zeros(batch_size, self.output).to(self.device)
    
        if type(input) is np.ndarray:
                input = torch.from_numpy(input).float().to(self.device)

        else:
            input = input.float().to(self.device)

        sum_out = None
        #x = x.view(self.batch_size, -1)
        for t in range(time_window):

            x_t = input[:,:,t]

            x_t = x_t.reshape(batch_size, -1)

            #layer forward
            h1_mem, h1_spike = mem_update(self.fc1, x_t, h1_mem, h1_spike)
            #h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            h4_mem, h4_spike = mem_update(self.fc4, h3_spike, h4_mem, h4_spike)
            h5_mem, h5_spike = mem_update(self.fc5, h4_spike, h5_mem, h5_spike)
                        
            sum_out = h5_spike if sum_out is None else sum_out + h5_spike

        #outputs = h2_sumspike / time_window
        return sum_out

############################################
# TODO: May be problems here
def r_stdp(pre_spike, post_spike, weight, reward, lr=0.03):
    '''
    pre_spike: (pre_neuron_num, timescale)
    post_spike: (post_neuron_num, timescale)
    layer.weight: (post_neuron_num, pre_neuron_num)
    '''
    

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


