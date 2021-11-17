import math
import logging
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
logger = logging.getLogger(__name__)
import torch.distributed as dist
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help='node rank for distributed training')
args = parser.parse_args()
dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
global_rank = dist.get_rank()

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 2e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = 'cere.pkl'
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model.cuda()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.loss_store = np.zeros(config.max_epochs)

        # state_dict = torch.load('cere.pkl',map_location="cuda:%d" % args.local_rank)  # 加载模型
        # self.model.load_state_dict(state_dict)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            train_dataset = self.train_dataset
            test_dataset = self.test_dataset
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            # train_iter = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True,shuffle=True,drop_last=True)  # 对于直接从硬盘读取数据num_workers=4 很重要 提高GPU利用率 linux
            train_iter = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True,shuffle=(train_sampler is None),sampler=train_sampler,drop_last=True)
            test_iter = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True,shuffle=(test_sampler is None),sampler=test_sampler,drop_last=True)
            
            if is_train:
                loader = train_iter
            else:
                loader = test_iter
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.cuda()  #dtype=torch.float64
                y = y.cuda()  #dtype=torch.float64

                # for i in range(20):
                #     m_x_spike = x[:,i,0]
                #     f_x_spike = x[:,i,1]
                #     u_x_spike = x[:,i,2]
                #     sigma_x_spike = x[:,i,3]

                #     opt_mx_spike = y[:,i,0]
                #     opt_ux_spike = y[:,i,1]
                m_x_spike = x[:,0]
                f_x_spike = x[:,1]
                u_x_spike = x[:,2]
                sigma_x_spike = x[:,3]

                opt_mx_spike = y[:,0]
                opt_ux_spike = y[:,1]

                # forward the model
                with torch.set_grad_enabled(is_train):
                    #sum_out, spike_out = model(m_x_spike,u_x_spike,sigma_x_spike,f_x_spike,time_window=50)
                    sum_out = model(m_x_spike,u_x_spike,sigma_x_spike,f_x_spike,time_window=50)
                    #loss
                    sum_mx = sum_out[:,0:4]
                    #spike_out_mx = spike_out[:,0:4,:]
                    sum_ux = sum_out[:,4:8]
                    opts_mx_torch = torch.sum(opt_mx_spike,dim=2)
                    opts_mx_torch = torch.sum(opts_mx_torch,dim=1)

                    opts_ux_torch = torch.sum(opt_ux_spike,dim=2) 
                    opts_ux_torch = torch.sum(opts_ux_torch,dim=1)                

                    #loss = torch.nn.MSELoss()(sum_mx,opts_mx_torch) + torch.nn.MSELoss()(sum_ux,opts_ux_torch)
                    loss = torch.nn.MSELoss()(torch.sum(sum_mx,dim=1),opts_mx_torch) + torch.nn.MSELoss()(torch.sum(sum_ux,dim=1),opts_ux_torch)

                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")#. lr {lr:e}

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                self.loss_store[epoch] = test_loss
                np.savetxt(f'loss_cere_loop_batch{config.batch_size}.txt',self.loss_store)
            

            # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #if epoch%50==0 :
            #    self.save_checkpoint(epoch)
            if args.local_rank == 0:
                if test_loss < best_loss :
                    best_loss=test_loss
                    print(test_loss)
                    #torch.save(self.model, "./model/model_large_best.pkl" )
                    self.save_checkpoint()
