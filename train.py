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
       
def _TrainOneEpoch(model, session, dataloader, cfg, epoch):
    lr = cfg.tr.lr * cfg.tr.lr_mapper.GetFactor(epoch)
    total_batch = len(dataloader)
    a_softmax = AverageMeter()
    a_center = AverageMeter()
    a_reg = AverageMeter()
    a_total = AverageMeter()
    a_acc = AverageMeter()

    # record current epoch number to the model, so that
    # when the model was loaded from a checkpoint file,
    # the model would know how many epoches it had runned
    session.run(model.op_assign_epoch, feed_dict={model.pl_epoch:epoch})
    
    for batch, (imgs, _, labels) in enumerate(dataloader):
        feed = {model.pl_images:imgs, model.pl_labels:labels, model.pl_phase:True, model.pl_lr:lr}
        target = [model.accuracy, model.loss_softmax, model.loss_center, model.loss_reg, model.op_train]
        result = session.run(target, feed_dict=feed)

        acc, l_softmax, l_center, l_reg, _ = result
        a_softmax.update(l_softmax, imgs.shape[0])
        a_center.update(l_center, imgs.shape[0])
        a_reg.update(l_reg, imgs.shape[0])
        total = l_softmax + l_center + l_reg
        a_total.update(total, imgs.shape[0])
        a_acc.update(acc, imgs.shape[0])
        
        if cfg.tr.verbose and batch % cfg.tr.print_every == 0:
            info = '[{0}][{1:3}/{2}]  acc:{3:6.3f}  softmax:{4:6.3f}  center:{5:6.3f}  reg:{6:6.3f}  total:{7:6.3f}'
            print(info.format(epoch, batch, total_batch, acc, l_softmax, l_center, l_reg, total))
    print('\n\n')
            
    auc = None
    if cfg.val.use_lfw and epoch % cfg.val.lfw_every == 0:
        lfw = LFWSet(cfg.val.lfw_pairs, cfg.val.lfw_dir, cfg.aug)
        dl = DataLoader(lfw, batch_size=16, shuffle=False)
        auc = ValidateAUC(model, session, dl, is_print=cfg.tr.verbose)
    
    acc_val = None
    if cfg.val.use_val and epoch % cfg.val.val_every == 0:
        data_val = TxtSet(cfg.val.path_root, cfg.val.path_txt, cfg.aug, processing='validation')
        dl = DataLoader(data_val, batch_size=16, shuffle=False)
        acc_val = ValidateACC(model, session, dl, is_print=cfg.tr.verbose)
        
    if cfg.sl.is_save and epoch % cfg.sl.save_every == 0:
        _SaveModel(session, model, cfg.sl.save_dir, epoch, is_print=cfg.tr.verbose)

    epoch_result = {}
    epoch_result['epoch'] = epoch
    epoch_result['softmax'] = a_softmax.avg
    epoch_result['reg'] = a_reg.avg
    epoch_result['center'] = a_center.avg
    epoch_result['total_loss'] = a_total.avg
    epoch_result['accuracy'] = a_acc.avg
    if auc is not None:
        epoch_result['auc'] = auc
    if acc_val is not None:
        epoch_result['acc_val'] = acc_val
    
    if cfg.tr.verbose:
        print('---------- average statistics for the whole epoch ----------')
        info = 'acc:{0:6.3f}  softmax:{1:6.3f}  center:{2:6.3f}  reg:{3:6.3f}  total:{4:6.3f}'
        print(info.format(epoch_result['accuracy'], epoch_result['softmax'], epoch_result['center'], epoch_result['reg'], epoch_result['total_loss']))
        print('\n\n')
        
    return epoch_result
        
def _Run(model, session, dataloader, cfg):
    
    current_epoch = session.run(model.epoch) + 1
    print('the training process starts from epoch {}'.format(current_epoch))
    results = []
    for epoch in range(current_epoch, cfg.tr.max_epoch):
       result = _TrainOneEpoch(model, session, dataloader, cfg, epoch)
       
       if result is not None:
           results.append(result)
       
    collected_results = {key: [result[key] for result in results] for key in results[0]}
    df = pd.DataFrame(collected_results)
    #columns = ['epoch', 'auc', 'accuracy', 'total_loss', 'softmax', 'center', 'reg']
    #df = df[columns]
    return df

    # validate accuracy performance on classification task
def ValidateACC(model, session, dataloader, is_print=True):
    acc = AverageMeter()
    softmax = AverageMeter()
    center = AverageMeter()
    reg = AverageMeter()
    total = AverageMeter()
    for imgs, _, labels in dataloader:
        size = imgs.shape[0]
        feed = {model.pl_images:imgs, model.pl_labels:labels, model.pl_phase:True}
        target = [model.accuracy, model.loss_softmax, model.loss_center, model.loss_reg]
        v_acc, v_soft, v_center, v_reg = session.run(target, feed_dict=feed)
        acc.update(v_acc, size)
        softmax.update(v_soft, size)
        center.update(v_center, size)
        reg.update(v_reg, size)
        total.update(v_soft + v_center + v_reg, size)
    
    if is_print:
        print('---------- validation on testing data ----------')
        info = 'acc:{0:6.3f}  softmax:{1:6.3f}  center:{2:6.3f}  reg:{3:6.3f}  total:{4:6.3f}'
        print(info.format(acc.avg, softmax.avg, center.avg, reg.avg, total.avg))
        print('\n\n')
        
    return acc.avg


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
    
    