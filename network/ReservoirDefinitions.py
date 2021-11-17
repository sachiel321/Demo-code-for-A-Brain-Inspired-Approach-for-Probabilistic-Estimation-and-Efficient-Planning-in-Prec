# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:14:16 2020

@author: anmol
"""
import numpy as np

np.random.seed(1)

#this file contains functions to create generate the Reservoir network

def create_random_reservoir(dims, frac_inhibitory, w_matrix, fanout):
    #dims : 3-D dimensions of the reservoir (each of dims[0], dims[1] and dims[2] should be 3 or greater)
    #frac_inhibitory : fraction of inhibitory nodes
    #w_matrix : 2x2 matrix of weights from Excitatory(E) to Inhibitory(I), E to E, I to E and I to I
    #fanout : number of nodes connected to each node
    
    n_nodes = dims[0]*dims[1]*dims[2]
    node_types = np.random.uniform(size=n_nodes)
    node_types[node_types>frac_inhibitory] = 1
    node_types[node_types<=frac_inhibitory] = -1
    adj_mat = np.zeros((n_nodes, n_nodes))
    all_connections = []
    all_weights = []
    for i in range(n_nodes):
        z = np.int32(i/(dims[0]*dims[1]))
        y = np.int32((np.int32(i) % np.int32(dims[0]*dims[1]))/dims[0])
        x = (np.int32(i) % np.int32(dims[0]*dims[1])) % np.int32(dims[0])
        
        #Assuming connectivity is limited to nxn cube surrounding the node (n is odd)
        conn_window = 3
        z_c = np.minimum(np.maximum(z,np.int32(conn_window/2)),dims[2]-1-np.int32(conn_window/2))
        y_c = np.minimum(np.maximum(y,np.int32(conn_window/2)),dims[1]-1-np.int32(conn_window/2))
        x_c = np.minimum(np.maximum(x,np.int32(conn_window/2)),dims[0]-1-np.int32(conn_window/2))
        choice_neighbors = np.random.choice(conn_window**3, fanout, replace=False)
        list_connected = []
        list_weights = []
        from_node_type = 1 - np.int32((node_types[i]+1)/2)
        for neighbor in choice_neighbors:
            z_loc = np.int32(neighbor/(conn_window**2)) + z_c - 1
            y_loc = np.int32((np.int32(neighbor) % (conn_window**2)) / conn_window) + y_c - 1
            x_loc = (np.int32(neighbor) % (conn_window**2)) % conn_window + x_c - 1
            neighbor_id = z_loc*dims[0]*dims[1] + y_loc*dims[0] + x_loc
            to_node_type = 1 - np.int32((node_types[neighbor_id]+1)/2)
            list_connected.append(neighbor_id)
            list_weights.append(w_matrix[from_node_type, to_node_type])
            adj_mat[i, neighbor_id] = w_matrix[from_node_type, to_node_type]
        all_connections.append(list_connected)
        all_weights.append(list_weights)
    return adj_mat, all_connections, all_weights
#w_mat = np.array([[3, 6],[-2, -2]])
#adj_mat, adj_list, weight_list = create_random_reservoir((5,5,5), 0.2, w_mat, 9)