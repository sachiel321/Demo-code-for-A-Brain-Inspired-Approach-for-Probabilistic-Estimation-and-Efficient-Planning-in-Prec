import argparse
import logging
import math

import numpy as np
import torch

from network.NetworkClasses_possion import EPropRSNU, SNN
from utils.coding_and_decoding import poisson_spike_multi

logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
logger = logging.getLogger(__name__)
#np.random.seed(1)

def get_parser():
    parser = argparse.ArgumentParser(description="LSM and SNU's pram")
    parser.add_argument("--N_step","-N",type=int,default=2)
    parser.add_argument("--load_model","-l",type=bool,default=False)
    parser.add_argument("--save_model","-s",type=bool,default=False)
    parser.add_argument("--lr","-lr",type=float,default=1e-4)
    parser.add_argument("--iter","-i",type=int,default=5000)
    parser.add_argument("--gpu",type=str,default='0')
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--label",type=str,default='train3test3')
    return parser

def poisson_spike(t, f, dt=0.1, dim=1):
    """ Generate a Poisson spike train.

    t: length
    f: frequency
    dt: time step; default 0.1ms
    """
    # dt, t in ms; f in Hz.
    return np.random.rand(dim, int(t / dt)) < (f * dt / 10)

class ScaledMSE(torch.nn.Module):
    def __init__(self):
        super(ScaledMSE, self).__init__()
        self.tiny = 1

    def forward(self,input,opts):
        if opts.cpu().item() == 0:
            temp = (input - opts)/self.tiny
            return torch.pow(temp,2)
        else:
            temp = (input - opts)/opts
            return torch.pow(temp,2)


parser = get_parser()
args = parser.parse_args()

def train_hippo(args=args):
    ###########Parameters##########
    N_step = args.N_step  # predict step
    load_model = args.load_model
    save_model = args.save_model
    learning_rate = args.lr

    dims = (7,7,7)
    bias = 4
    n_in = dims[0]*dims[1]*dims[2]
    w_mat = 4*20*np.array([[3, 6],[-2, -2]])
    steps = 50
    dt = 0.1
    ch = 5 #input num
    possion_num = 50
    best_loss = 10
    batch_size = args.batch_size
    counter = np.zeros([batch_size,1]).astype(np.float32)
    
    ################################

    #load networks
    gpu = args.gpu
    device = 'cuda:'+gpu
    if load_model == False:
        reservoir_network = EPropRSNU(ch*bias,n_in)
        snu = SNN(batch_size=args.batch_size,input_size=n_in,num_classes=bias, encoding_num=bias,possion_num=possion_num,gpu=gpu)
        reservoir_network = reservoir_network.to(device)
        snu = snu.to(device)

    else:
        print( "loading model from " + "my_snu_model")
        reservoir_network = torch.load(f"my_lsm_model_2.pkl")
        snu = torch.load(f"my_snu_model_2.pkl", map_location='cuda:'+gpu)
        snu.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            snu = snu.to(device)
            reservoir_network = reservoir_network.to(device)
        for k, m in snu.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability

        print("load model successfully")

    #init matrix
    v_h_init = torch.zeros([batch_size,n_in]).to(device)
    snu_output_np = np.zeros([batch_size, N_step, bias])
    opts_np = np.zeros((batch_size, N_step, bias))

    criterion = torch.nn.MSELoss()
    #criterion = ScaledMSE()
    optimizer = torch.optim.AdamW([{'params':reservoir_network.parameters()},{'params':snu.parameters()}], lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[300,600],gamma = 0.5)

    loss_store = []

    for i in range(args.iter):
        if i % 50 == 0 and i > 0:
            print(i)
            if save_model == True:
                torch.save(snu,f"my_snu_model_{args.N_step}.pkl")
                torch.save(reservoir_network,f"my_lsm_model_{args.N_step}.pkl")
                print("saved model successfully")
        #create data
        delt_z = np.random.rand(batch_size).reshape(batch_size,1)*50
        # delt_z = counter+np.random.rand(batch_size).reshape(batch_size,1)*0.01

        # idx50plus = np.where(counter>=50)
        # idx50minus = np.where(counter<50)
        # counter[idx50plus] = 0
        # counter[idx50minus] = counter[idx50minus] + 1

        ipts = torch.zeros([batch_size, 5])
        ipts[:,0] = torch.from_numpy(delt_z[:,0])

        opts_ones = np.array([0.7189618,1.287569,1.3333306,0.5672474]).reshape(1,4)
        opts = delt_z@opts_ones
        opts = np.array([opts*(i+1) for i in range(5)]).swapaxes(0,1)

        #opts = np.log(opts)/np.log(1.04)

        # ipts = ipts.astype(np.int32)
        # opts = opts.astype(np.int32)

        temp_loss = None

        for iteration in range(N_step):
            
            train_in_spikes = torch.from_numpy(poisson_spike_multi(t=possion_num*dt,f=ipts, dim=bias)).float().to(device).reshape(batch_size,-1,possion_num)
            
            #LSM mode
                
            rate_coding = reservoir_network.forward(train_in_spikes, v_h_init)
            
            rate_coding = rate_coding.permute(1,0,2)

            #SNU mode
            snu_output = snu.forward(input=rate_coding,task="LSM",time_window=possion_num)

            temp_opts_torch = torch.from_numpy(opts[:,iteration,:]).float().to(device)

            #build next step data
            for m in range(4):
                snu_output_np[:,iteration,m] = torch.sum(snu_output[:,4*m:4*m+4],dim=1).cpu().detach().numpy()
                opts_np[:,iteration,m] = temp_opts_torch[:,m].cpu().numpy()
            
            
            ipts,_,__ = np.split(ipts,[1,1],axis = 1) # get delt z
            ipts = np.hstack((ipts,snu_output_np[:,iteration])) #t+1 input

            mu_x = torch.sum(snu_output[:,0:bias])
            mu_x_opt = temp_opts_torch[:,0]

            mu_x_opt = torch.clamp(mu_x_opt,0,50*bias).float()

            mu_y = torch.sum(snu_output[:,bias:bias*2])
            mu_y_opt = temp_opts_torch[:,1]

            mu_y_opt = torch.clamp(mu_y_opt,0,50*bias).float()

            sigma_x = torch.sum(snu_output[:,bias*2:bias*3])
            sigma_x_opt = temp_opts_torch[:,2]

            sigma_x_opt = torch.clamp(sigma_x_opt,0,50*bias).float()

            sigma_y = torch.sum(snu_output[:,bias*3:bias*4])
            sigma_y_opt = temp_opts_torch[:,3]

            sigma_y_opt = torch.clamp(sigma_y_opt,0,50*bias).float()


            #loss func
            loss_mu_x = criterion(mu_x, mu_x_opt)
            loss_mu_y = criterion(mu_y, mu_y_opt)
            loss_sigma_x = criterion(sigma_x, sigma_x_opt)
            loss_sigma_y = criterion(sigma_y, sigma_y_opt)
            loss = loss_mu_x + loss_mu_y + loss_sigma_x + loss_sigma_y

            logger.info(
                f"episode:{i} it:{iteration} loss:{loss.cpu().item()},{loss_mu_x.cpu().item()} ,{loss_mu_y.cpu().item()}, {loss_sigma_x.cpu().item()}, {loss_sigma_y.cpu().item()}", 
            )   

            loss_store.append(loss.item())
            np.savetxt(f'loss_{args.N_step}_{args.label}.txt',loss_store,fmt='%0.8f')
            np.save(f'output_{args.N_step}_{args.label}.npy',snu_output_np)
            np.save(f'opts_{args.N_step}_{args.label}.npy',opts_np)
            #save the best model
            if loss.item() < best_loss and i > 10:
                best_loss = loss.item()
                if save_model == True:
                    torch.save(snu,f"my_best_snu_model_{args.N_step}.pkl")
                    torch.save(reservoir_network,f"my_best_lsm_model_{args.N_step}.pkl")
                    print("saved best model successfully with loss",best_loss)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reservoir_network.reset()
