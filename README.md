# Demo-code-for-A-Brain-Inspired-Approach-for-Probabilistic-Estimation-and-Efficient-Planning-in-Prec
Demo code for our paper &lt;A Brain-Inspired Approach for Probabilistic Estimation and Efficient Planning in Precision Physical >

## Supported Environments
Ubuntu 16.04.7 LTS


## Deployment steps


1. Install environment dependencies.

```
pip install -r requirements.txt
```


2. For supervised learning demo, run <train_sl.py> in the following ways.\
Choose variable "**brain_areas**" value first in <**train_sl.py**>.\
For the cerebellum network and the prefrontal network:

```
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 train_sl.py
```

For the hippocampus network:
    
```
python train_sl.py --save_model --label demo
```

3. For reinforcement learning demo, run <train_combine_rl.py> in the fllowing ways.\
Configuring hyperparameters in <train_combine_rl.py> firstly, then run the demo:
        
```
python train_combine_rl.py
```



## Directory Structure Description

```
.
├── data                                            # The data used for supervised learning
│   ├── data_cere_withuxfiltered.npy
│   ├── data_pre.npy
│   └── data_process.py
├── network
│   ├── __init__.py
│   ├── NetworkClasses_possion.py                   # Core network implementation
│   ├── NetworkClasses.py # Temporarily useless
│   ├── ObjectClasses.py
│   └── ReservoirDefinitions.py
├── requirements.txt
├── train_combine_rl.py                             # Entry file for reinforcement learning
├── trainer
│   ├── __init__.py
│   ├── agent.py
│   ├── cerebellum.py
│   ├── combine.py
│   ├── hippocampus.py
│   └── prefrontal.py
├── train_sl.py                                     # Entry file for supervised learning
└── utils
    ├── __init__.py
    ├── buffer.py
    ├── coding_and_decoding.py
    ├── evaluate.py
    ├── integral.py
    ├── num_in_out.py
    └── simulation.py

```


