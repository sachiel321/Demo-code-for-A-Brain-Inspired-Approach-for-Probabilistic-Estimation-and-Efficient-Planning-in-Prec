import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from network.NetworkClasses_possion import cere_SNN, prefrontal_model
from utils.coding_and_decoding import seed_everything

seed_everything(42)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

brain_areas = 'hippo' # The model to train. Range from ['hippo', 'cere', 'pre']
division_constant = 0.8 # How many samples to train. Range from (0,1)
possion_num = 50 # The number of spikes encoding a sample. Must be positive integer 
batch_size = 256
lr = 3e-4 # learning rate. Recommend range [1e-4,5e-3]

class MyCereDataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __getitem__(self, index):
        data_step = self.data[index]
        x = torch.tensor(data_step[0:4], dtype=torch.float)
        y = torch.tensor(data_step[4:], dtype=torch.float)
        return x,y

    def __len__(self):
        return self.data.shape[0]

class MyPrefrontalParaDataset(Dataset):
    def __init__(self,root):
        file_box = []
        list = os.listdir(root) #文件名
        for i in range(0,len(list)):
            com_path = os.path.join(root,list[i])
            file_box.append(com_path)
        self.list_files = np.array(file_box)

    def __getitem__(self, index):
        data_step = np.load(self.list_files[index])
        x = torch.tensor(data_step[0:300,0:17], dtype=torch.float)  #训练不迭代 同时进行  测试需要迭代
        y = torch.tensor(data_step[1:301,16], dtype=torch.float)    #49,9
        return x,y

    def __len__(self):
        return len(self.list_files)

if brain_areas == 'cere':
    data =np.load("data_cere_withuxfiltered.npy").reshape(-1,6)
    print(data.shape)

    data_cere_spike = np.zeros([data.shape[0],6,4,possion_num])
    data_cere_spike[:,0,:,:] = (np.random.rand(data.shape[0],4,possion_num) < (data[:,0].reshape(data.shape[0],1,1)+6)/12).astype(np.int)
    data_cere_spike[:,1,:,:] = (np.random.rand(data.shape[0],4,possion_num) < (data[:,1].reshape(data.shape[0],1,1))/400).astype(np.int)
    data_cere_spike[:,2,:,:] = (np.random.rand(data.shape[0],4,possion_num) < (data[:,2].reshape(data.shape[0],1,1))/40).astype(np.int)
    data_cere_spike[:,3,:,:] = (np.random.rand(data.shape[0],4,possion_num) < (data[:,3].reshape(data.shape[0],1,1))/70).astype(np.int)
    data_cere_spike[:,4,:,:] = (np.random.rand(data.shape[0],4,possion_num) < (data[:,4].reshape(data.shape[0],1,1)+6)/12).astype(np.int)
    data_cere_spike[:,5,:,:] = (np.random.rand(data.shape[0],4,possion_num) < (data[:,5].reshape(data.shape[0],1,1)+4)/400).astype(np.int)

    division = int(division_constant * data_cere_spike.shape[0])
    train = data_cere_spike[:division]
    test = data_cere_spike[division:]

    train_dataset = MyCereDataset(train)
    test_dataset = MyCereDataset(test)

    cere_batch_size = batch_size

    model = cere_SNN(batch_size=cere_batch_size,num_in_MF=1,num_out_MF=16,num_out_GC=1000,num_out_PC=400,num_out_DCN=1,possion_num=possion_num)
    tokens_per_epoch = len(train_dataset) * 70
    train_epochs = 1000 # todo run a bigger model and longer, this is tiny
    # initialize a trainer instance and kick off training
    from trainer.cerebellum import Trainer, TrainerConfig  # train
    tconf = TrainerConfig(max_epochs=1000, batch_size=cere_batch_size, learning_rate=lr,lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,ckpt_path=f'cere_batch{cere_batch_size}.pkl')
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()

elif brain_areas == 'pre':

    train = './PreDataTrain'
    test = './PreDataTest'

    train_dataset = MyPrefrontalParaDataset(train)
    test_dataset = MyPrefrontalParaDataset(test)

    pre_batch_size = batch_size

    model = prefrontal_model(batch_size=pre_batch_size, num_hidden1=256, num_hidden2=512, num_hidden3=128,N_step=3)

    tokens_per_epoch = len(train_dataset) * 70
    train_epochs = 1000 # todo run a bigger model and longer, this is tiny
    from trainer.prefrontal import Trainer, TrainerConfig  # train
    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=train_epochs, batch_size=pre_batch_size, learning_rate=lr,lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,ckpt_path='pre_batch{cere_batch_size}.pkl')
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()

elif brain_areas == 'hippo':
    from trainer.hippocampus import train_hippo
    # See training parameters for the hippocampus network in trainer.hippocampus.py
    train_hippo()
