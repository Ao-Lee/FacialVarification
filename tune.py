# -*- coding: utf-8 -*-
import numpy as np

try:
    from data import TrainingSet
    from UtilsData import DataLoader
    from UtilsConfig import Cfg
    from train import TrainModel
except ImportError:
    from .data import TrainingSet
    from .UtilsData import DataLoader
    from .UtilsConfig import Cfg
    from .train import TrainModel

def _SetRandomCfg(cfg):
    # learning rate:
    # 0.005 - 0.0005, log space
    lr_max = -6
    lr_min = -10
    lr = np.exp(np.random.uniform(low=lr_min, high=lr_max))
    cfg.SetArgs(cfg.train, lr=lr)
    
    # regularization:
    # 1e-3 - 1e-6, log space
    re_max = -4
    re_min = -14
    reg = np.exp(np.random.uniform(low=re_min, high=re_max))
    cfg.SetArgs(cfg.train, reg=reg)
    
    # weight initialization:
    # 0.01 - 1, log space
    std_max = -1
    std_min = -4
    std = np.exp(np.random.uniform(low=std_min, high=std_max))
    std = 'xavier' if np.random.random>0.5 else std
    cfg.SetArgs(cfg.train, stddev=std)
    dist = 'uniform' if np.random.random>0.5 else 'normal'
    cfg.SetArgs(cfg.train, distribution=dist)
    
    # lambda 
    # 0.01 - 1, log space
    lmbda_max = -1
    lmbda_min = -4
    lmbda = np.exp(np.random.uniform(low=lmbda_min, high=lmbda_max))
    cfg.SetArgs(cfg.model, lmbda=lmbda)
    
    # alpha
    # 0.1 - 0.9, uniform space
    alpha_max = 0.9
    alpha_min = 0.1
    alpha = np.random.uniform(low=alpha_min, high=alpha_max)
    cfg.SetArgs(cfg.model, alpha=alpha)
    
    # others
    cfg.SetArgs(cfg.model, lmbda=0.01, optimizer='adam', loss_method='center')
    cfg.SetArgs(cfg.train, is_save=False, is_load=False)
    cfg.SetArgs(cfg.train, max_epoch=10)
    cfg.SetArgs(cfg.train, verbose=False)
    
    
def Tune():
    infos = []
    cfgs = []
    iteration = 10
    cfg_tmp = Cfg()
    training_set = TrainingSet(cfg_tmp.train.data_dir, cfg_tmp.augmentation)
    dataloader = DataLoader(training_set, batch_size=cfg_tmp.train.batch_size, shuffle=True)
    n_classes = len(training_set.mapping)
    for i in range(iteration):
        print('hyper param tunning, iteration {}/{}'.format(i+1, iteration))
        cfg = Cfg()
        cfg.model.n_classes = n_classes
        cfg = _SetRandomCfg(cfg)
        info = TrainModel(cfg, dataloader)
        infos.append(info)
        cfgs.append(cfg)
    return infos, cfgs

if __name__=='__main__':
    infos, cfgs = Tune()

    
