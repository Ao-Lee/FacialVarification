# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse

class InfoGenerator():
        
    def _GetDefaultContent(self):
        assert False, 'must be implemented by subclasses'
        
    def GetContent(self, **kwargs):
        cfg = self._GetDefaultContent()
        for name, value in kwargs.items():
            assert hasattr(cfg, name), 'argument {} not defined'.format(name)
            setattr(cfg, name, value)
        return cfg
    
class LogGenerator(InfoGenerator):
    def _GetDefaultContent(self):
        parser = argparse.ArgumentParser()
        
        info_iteration = 'current iteration'
        info_acc_tr = 'accuracy on current training batch'
        info_loss_softmax_tr = 'softmax loss on current training batch'
        info_loss_re_tr = 'regularization loss on current training batch'
        info_loss_tr = 'total loss on current training batch'
        info_acc_val = 'accuracy on validation set'
        info_loss_softmax_val = 'softmax loss on validation set'
        info_loss_re_val = 'regularization loss on validation set'
        info_loss_val = 'total loss on validation set'
        
        parser.add_argument('--iteration',      type=int,       help=info_iteration,    default=None)
        parser.add_argument('--acc_tr',         type=float,     help=info_acc_tr,       default=None)
        parser.add_argument('--loss_softmax_tr',type=float,     help=info_loss_softmax_tr, default=None)
        parser.add_argument('--loss_re_tr',     type=float,     help=info_loss_re_tr,   default=None)
        parser.add_argument('--loss_tr',        type=float,     help=info_loss_tr,      default=None)
        parser.add_argument('--acc_val',        type=float,     help=info_acc_val,      default=None)
        parser.add_argument('--loss_softmax_val', type=float,   help=info_loss_softmax_val, default=None)
        parser.add_argument('--loss_re_val',    type=float,     help=info_loss_re_val,  default=None)
        parser.add_argument('--loss_val',       type=float,     help=info_loss_val,     default=None)
        return parser.parse_args([])
        
class ReportGenerator(InfoGenerator):
    def _GetDefaultContent(self):
        parser = argparse.ArgumentParser()
        
        info_acc = 'validation accuracy. '
        info_acc += 'Since accuracy may decrease during training, this accuracy may not be the best acc'
        info_best_acc = 'highest validation accuracy ever in the training history'
        info_loss_softmax = 'softmax loss on validation dataset'
        info_loss_re = 'regularization loss on validatin dataset'
        info_loss = 'total loss on validation dataset'
        info_traing_loss_hist = 'training losses for each batch'
        info_distribution = 'initial weight distribution of conv and fc layers'
        info_reg = 'regularization strength'
        info_lr = 'initial learning rate'
        
        parser.add_argument('--acc',            type=float,     help=info_acc,          default=None)
        parser.add_argument('--best_acc',       type=float,     help=info_best_acc,     default=None)
        parser.add_argument('--loss_softmax',   type=float,     help=info_loss_softmax, default=None)
        parser.add_argument('--loss_re',        type=float,     help=info_loss_re,      default=None)
        parser.add_argument('--loss',           type=float,     help=info_loss,         default=None)
        parser.add_argument('--traing_loss_hist', type=list, help=info_traing_loss_hist,default=None)
        parser.add_argument('--distribution',   type=str,       help=info_distribution, default=None)
        parser.add_argument('--reg',            type=float,     help=info_reg,          default=None)
        parser.add_argument('--lr',             type=float,     help=info_lr,           default=None)
        return parser.parse_args([])
        
class ModelCfgGenerator(InfoGenerator):
    def _GetDefaultContent(self):
        parser = argparse.ArgumentParser()
        
        info_n_classes = 'number of classes in the classification problem'
        info_distribution = 'uniform or normal distribution for weight initialization'
        choices_distribution = ['uniform', 'normal']
        info_seed = 'random seed for weight initialization and dropout'
        info_seed += 'set seed to achieve repreducable results'
        info_reg = 'regularization for weights of cov layers and fc layers. Biases are excluded'
        info_bn = 'set to True to use batch norm'
        info_loss = 'loss methord to use. There are two options: softmax and center'
        choices_loss = ['softmax', 'center']
        info_alpha = 'learning rate of centers. only used when set loss to center. '
        info_alpha += 'for more details, please check the paper <A Discriminative Feature Learning Approach for Deep Face Recognition>'
        info_lambda = 'center softmax tradeoff. only used when set loss to center. '
        info_lambda += 'note that the config name is lmbda not lambda, since lambda is a fuckin python key word'
            
        parser.add_argument('--n_classes',      type=int,       help=info_n_classes,        default=10)
        parser.add_argument('--distribution',   type=str,       help=info_distribution,     default='uniform', choices=choices_distribution)
        parser.add_argument('--reg',            type=float,     help=info_reg,              default=0.0001)
        parser.add_argument('--use_bn',         type=bool,      help=info_bn,               default=True)
        parser.add_argument('--seed',           type=int,       help=info_seed,             default=231)
        parser.add_argument('--loss',           type=str,       help=info_loss,             default='center', choices=choices_loss)
        parser.add_argument('--alpha',          type=float,     help=info_alpha,            default=0.5)
        parser.add_argument('--lmbda',          type=float,     help=info_lambda,           default=0.1)
        return parser.parse_args([])
    
class TrainCfgGenerator(InfoGenerator):
    def _GetDefaultContent(self):
        data_dir = 'F:\\Dropbox\\DataScience\\FacialVarificationProject\\data\\MNIST_data'
        save_dir = 'F:\\model\\BigModel'
        load_dir = 'F:\\model'
        
        parser = argparse.ArgumentParser()
        # General configuration   
        info_data_dir = 'location of training data'
        info_lr = 'initial learning rate. Later, there will be a mapping system between learning rate and epoch'
        info_max_batch = 'how many batch to train'
        info_verbose = 'if true, show details during training'
        info_print_every = 'print training infomation repeatedly after print_every iteration'
        info_mapper = 'dynamic learning rate.'
        mapper = LearningRateMapper()
        
        parser.add_argument('--data_dir',       type=str,       help=info_data_dir,     default=data_dir)
        parser.add_argument('--batch_size',     type=int,       help='',                default=256)
        parser.add_argument('--lr',             type=float,     help=info_lr,           default=0.0001)
        parser.add_argument('--max_batch',      type=int,       help=info_max_batch,    default=5000)
        parser.add_argument('--verbose',        type=bool,      help=info_verbose,      default=False)
        parser.add_argument('--print_every',    type=int,       help=info_print_every,  default=200)
        parser.add_argument('--lr_mapper', type=LearningRateMapper, help=info_mapper,   default=mapper)
        
        
        # saving & loading configuration
        info_is_save = 'if true, save the session to checkpoint file after save_every iteration'
        info_save_every = 'save the session to disk after save_every iteration if is_save is set to true'
        info_save_every += 'if set is_save to False, this term will be ignored'
        info_save_dir = 'where do you want to save the model. Ignored if is_save is set to False'
        info_is_load = 'if true, initialize the session from checkpoint file'
        info_load_dir = 'where do you want to load the model. Ignored if is_load is set to False'

        parser.add_argument('--is_save',        type=bool,      help=info_is_save,      default=False)
        parser.add_argument('--save_every',     type=int,       help=info_save_every,   default=500)
        parser.add_argument('--save_dir',       type=str,       help=info_save_dir,     default=save_dir)
        parser.add_argument('--is_load',        type=bool,      help=info_is_load,      default=False)
        parser.add_argument('--load_dir',       type=str,       help=info_load_dir,     default=load_dir)
        return parser.parse_args([])

# implementation of dynamic learning rate
# this class maps each iteration to a decay factor
class LearningRateMapper():
    def __init__(self):
        self._iteration = []
        self._decay = []
        self._Schedule()
        self._Compute()
        
    def _Add(self, iteration, decay_factor):
        self._iteration.append(iteration)
        self._decay.append(decay_factor)
        
    def _Schedule(self):
        self._Add(iteration=0,      decay_factor=1)
        #self._Add(iteration=600,   decay_factor=0.1)
        
    def _Compute(self):
        self.data = pd.DataFrame([])
        self.data['iter'] = pd.Series(self._iteration)
        self.data['decay'] = pd.Series(self._decay)
        self.data.sort_values(by='iter', ascending=True, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
    def GetFactor(self, iteration):
        which = np.sum(self.data['iter'] <= iteration) - 1
        return self.data.loc[which, 'decay']
        
   