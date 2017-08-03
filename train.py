# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import pandas as pd

try:
    from data import TxtSet, LFWSet
    from UtilsData import DataLoader
    from UtilsConfig import Cfg
    from UtilsModel import Inception_Res_V1
    from validate import ValidateACC, ValidateAUC
    from avg import AverageMeter
except ImportError:
    from .data import TxtSet, LFWSet
    from .UtilsData import DataLoader
    from .UtilsConfig import Cfg
    from .UtilsModel import Inception_Res_V1
    from .validate import ValidateACC, ValidateAUC
    from .avg import AverageMeter
    

        
def _SaveModel(session, model, path, epoch, is_print=True):
    if is_print:
        print('---------- Serialization info ----------')
        print('begin to save model to {}'.format(path))
    model.saver.save(session, path, epoch)
    if is_print:
        print('model successfully saved')
       


def TrainModel(cfg, dataloader):
    g = tf.Graph()
    with g.as_default():
        np.random.seed(seed=cfg.tr.seed)
        random.seed(cfg.tr.seed)
        tf.set_random_seed(cfg.tr.seed)
        model = Inception_Res_V1(cfg=cfg.model)
        session_cfg =tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        session = tf.Session(config=session_cfg)
  
        if cfg.sl.is_load:
            print('begin to load model from file')
            ckpt = tf.train.get_checkpoint_state(cfg.sl.load_dir)
            assert ckpt and ckpt.model_checkpoint_path, 'unable to load checkpoint'
            model.saver.restore(session, ckpt.model_checkpoint_path)
            print('model sucessfully loaded')
        else:
            print('begin to generate a new model, the model will be trained from scratch')
            session.run(model.op_init)
            print('model sucessfully created')
            
        info = _Run(model, session, dataloader, cfg)
        session.close()
    return info

def _SetCfg(cfg):
    cfg.SetArgs(cfg.model, reg=0.00005, optimizer='adam', loss_method='softmax')
    cfg.SetArgs(cfg.tr, lr=0.01)
    cfg.SetArgs(cfg.model, lmbda=0.1)
    cfg.SetArgs(cfg.sl, is_save=True, save_every=5, is_load=False)
    cfg.SetArgs(cfg.tr, max_epoch=30)
    return cfg

if __name__ == '__main__':
    cfg = Cfg()
    data_tr = TxtSet(cfg.tr.path_root, cfg.tr.path_txt, cfg.aug, processing='training')
    dataloader = DataLoader(data_tr, batch_size=cfg.tr.batch_size, shuffle=True)
    cfg.SetArgs(cfg.model, n_classes=data_tr.n_classes)
    cfg = _SetCfg(cfg)
    info = TrainModel(cfg, dataloader)
    
'''
(1)下载数据集
(3)使用triplet loss试一试
(4)用classification做超参数搜索
'''
    
    