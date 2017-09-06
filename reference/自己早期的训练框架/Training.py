# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ModelUtils
import ConfigUtils

def _SubtractByMean(x):
    # subtract x by the mean of each img
    mean = np.average(x, axis=1).reshape(-1, 1)   #(Batch_size, 1)
    result = x - mean
    return result
        
def _Notice():
    print('tr_acc和tr_loss是按照一个Batch计算的，有较大波动很正常')
    print('val_acc和val_loss是基于整个validation data set计算的')
        
def TrainModel(model, session, mnist, cfg):
    data_tr = mnist.train
    data_val = mnist.validation
    loss_history = []
    logs = []
    acc_val_best_so_far = 0
    current_step = session.run(model.global_step)
    lg = ConfigUtils.LogGenerator()
    for iteration in range(current_step, cfg.max_batch+1):
        batch = data_tr.next_batch(cfg.batch_size)
        learn_rate = cfg.lr * cfg.lr_mapper.GetFactor(iteration)
        feed_dict = {model.pl_images:_SubtractByMean(batch[0]), model.pl_labels:batch[1], model.pl_dropout_rate:0.5, model.pl_phase:1, model.pl_lr:learn_rate}
        loss, _ = session.run([model.loss, model.op_train], feed_dict=feed_dict)
        loss_history.append(loss)
        
        if cfg.is_save and iteration % cfg.save_every==0:
            model.saver.save(session, cfg.save_dir, iteration)
        if iteration % cfg.print_every != 0:
            continue
        feed_dict[model.pl_dropout_rate] = 0
        feed_dict[model.pl_phase] = 0
        acc_tr, loss_softmax_tr, loss_re_tr = session.run([model.accuracy, model.loss_softmax, model.loss_reg], feed_dict=feed_dict)
        acc_val, loss_softmax_val, loss_re_val = EvaluateModel(model, session, data_val, cfg)
        
        acc_val_best_so_far = acc_val if acc_val > acc_val_best_so_far else acc_val_best_so_far
        log = lg.GetContent()
        log.iteration = iteration
        log.acc_tr = acc_tr
        log.loss_softmax_tr = loss_softmax_tr
        log.loss_re_tr = loss_re_tr
        log.loss_tr = loss_softmax_tr + loss_re_tr
        log.acc_val = acc_val
        log.loss_softmax_val = loss_softmax_val
        log.loss_re_val = loss_re_val
        log.loss_val = loss_softmax_val + loss_re_val
        logs.append(log)
        if cfg.verbose:
            print('iteration: {0}\t acc_tr: {1:4.3f}\t acc_te: {2:4.3f}\t loss_tr: {3:4.4f}\t loss_val: {4:4.4f}\t lr: {5:4.2f}'.format(iteration, acc_tr, acc_val, log.loss_tr, log.loss_val, np.log(learn_rate)))
            
    acc_val, loss_softmax_val, loss_re_val = EvaluateModel(model, session, data_val, cfg) 
    rg = ConfigUtils.ReportGenerator()
    report = rg.GetContent()
    report.acc = acc_val
    report.best_acc = acc_val_best_so_far
    report.loss_softmax =loss_softmax_val
    report.loss_re = loss_re_val
    report.loss = loss_softmax_val + loss_re_val
    report.traing_loss_hist = loss_history
    report.distribution = model.cfg.distribution
    report.reg = model.cfg.reg
    report.lr = cfg.lr
    return report, logs

'''
evaluate model on test or validation set
returns a tuple containing:
    (1) validation accuracy
    (2) validation loss (cross entropy part)
    (3) validation loss (regularization)
''' 
def EvaluateModel(model, session, data, cfg):
    iteration = data.num_examples // cfg.batch_size
    list_acc = []
    list_loss_softmax = []
    list_loss_reg = []
    for _ in range(iteration):
        batch = data.next_batch(batch_size=cfg.batch_size, shuffle=False)
        feed_dict = {model.pl_images:_SubtractByMean(batch[0]), model.pl_labels:batch[1], model.pl_dropout_rate:0, model.pl_phase:0}
        acc, loss_softmax, loss_reg = session.run([model.accuracy, model.loss_softmax, model.loss_reg], feed_dict=feed_dict)
        list_acc.append(acc)
        list_loss_softmax.append(loss_softmax)
        list_loss_reg.append(loss_reg)
    return np.average(list_acc), np.average(list_loss_softmax), np.average(list_loss_reg)
    
def Run(cfg_model, cfg_train, mnist):
    g = tf.Graph()
    with g.as_default():
        model = ModelUtils.SmallModel(cfg=cfg_model)
        session = tf.InteractiveSession()
        if cfg_train.is_load:
            ckpt = tf.train.get_checkpoint_state(cfg_train.load_dir)
            assert ckpt and ckpt.model_checkpoint_path, 'unable to load check point'
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            session.run(model.op_init)

        report, logs = TrainModel(model, session, mnist, cfg_train)
        session.close()
    #node_num = len(tf.get_default_graph().as_graph_def().node)
    #print('there are {} nodes in current graph'.format(node_num))
    return report, logs

def Reports2Df(list_report):
    attrs = ['reg','lr','distribution','acc','best_acc','loss','loss_softmax','loss_re']
    df = pd.DataFrame([])
    for attr in attrs:
        values = []
        for report in list_report:
            value = getattr(report, attr)
            values.append(value)
        if attr=='reg' or attr=='lr':
            values = np.log(values)
        df[attr] = pd.Series(values)
    return df
    
def Logs2Df(logs):
    attrs = ['iteration','acc_tr','acc_val','loss_tr','loss_val']
    df = pd.DataFrame([])
    for attr in attrs:
        values = []
        for log in logs:
            value = getattr(log, attr)
            values.append(value)
        df[attr] = pd.Series(values)
    return df
    
def Tune():
    '''
    for small net
    re_max = -4
    re_min = -14
    lr_max = -6
    lr_min = -10
    '''
    re_max = -4
    re_min = -14
    lr_max = -6
    lr_min = -10
    distribution = np.array(['uniform', 'normal'])
    iteration = 1
    max_batch = 1000
    verbose = True
    
    #list_lr = np.exp(np.linspace(start=-9, stop=-13, num=iteration))
    list_lr = np.exp(np.random.uniform(low=lr_min, high=lr_max, size=iteration))
    list_re = np.exp(np.random.uniform(low=re_min, high=re_max, size=iteration))
    mask_distribution = np.random.randint(low=0, high=len(distribution), size=iteration)
    list_dstribution = distribution[mask_distribution]

    
    _Notice()
    list_report = []
    list_logs = []
    mcg = ConfigUtils.ModelCfgGenerator()
    ccg = ConfigUtils.TrainCfgGenerator()
    
    for i in range(iteration):
        print('number of hyperparam setting tried: {}/{}'.format(i+1, iteration))
        cfg_model = mcg.GetContent(distribution=list_dstribution[i], reg=list_re[i])
        cfg_train = ccg.GetContent(lr=list_lr[i], max_batch=max_batch, verbose=verbose)
        mnist = input_data.read_data_sets(cfg_train.data_dir , one_hot=False, seed=231)
        report, logs = Run(cfg_model,cfg_train, mnist)
        log_df = Logs2Df(logs)
        list_report.append(report)
        list_logs.append(log_df)
      
    df = Reports2Df(list_report)
    return df, list_logs

if __name__ == '__main__':
    df, logs = Tune()
    result = df.sort_values(by='acc', ascending=False)
    result.reset_index(drop=True, inplace=True)
    #loss = list_histories[0]
    #plt.plot(np.arange(len(loss)), loss)



