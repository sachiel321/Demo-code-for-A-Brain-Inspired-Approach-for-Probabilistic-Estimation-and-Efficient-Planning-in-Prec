# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:47:19 2020

@author: anmol
"""
import numpy as np

np.random.seed(1)

class Spike:
    
    def __init__(self, spike_values, receiver_nodes, receive_times):
        self.receiver_nodes = receiver_nodes
        self.receive_times = receive_times
        self.spike_values = spike_values

class Neuron:
    V_rest = 0
    V_th = 50
    
    def __init__(self, sid, out_conn, out_weights, fanout, tau, t_ref, spike_propagation_base_time=None):
        self.tau = tau
        self.t_ref = t_ref
        self.sid = sid
        self.Vp = self.V_rest
        self.out_conn = out_conn
        self.fanout = fanout
        self.last_spike = -15
        self.last_receive_spike = 0
        self.out_weights = out_weights
        if spike_propagation_base_time == None:
            self.spike_propagation_time = [10 - np.random.randint(5) for i in range(fanout)] #time taken for spike to propaate 
        else:
            self.spike_propagation_time = [spike_propagation_base_time - np.random.randint(np.int32((spike_propagation_base_time/2))) for i in range(fanout)]

    def send_spike(self, time):
        send_times = [time + self.spike_propagation_time[i] for i in range(len(self.spike_propagation_time))]
        spike = Spike(self.out_weights, self.out_conn, send_times)
        self.last_spike = time
        self.Vp = self.V_rest
        return spike
    
    def receive_spike(self, time, value):
        spiked = False
        if (time - self.last_spike) > self.t_ref:
            V = self.Vp * np.exp(-(time-self.last_receive_spike)/self.tau)
            V = V + value
            self.last_receive_spike = time
            if V>self.V_th:
                spiked = True
                spike = self.send_spike(time)
            else:
                self.Vp = V
        if spiked:
            return spike
        else:
            return None
    
    def reset_spike(self):
        self.Vp = self.V_rest
        self.last_spike = -15
        self.last_receive_spike = 0
        return None
