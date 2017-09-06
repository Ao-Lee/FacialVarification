# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np

   
class Cfg():
    def __init__(self):
        self.tr = self._GetTrainParser()
        self.aug = self._GetAugmentationParser()      
        self.model = self._GetModelParser()
        self.val = self._GetValidationParser()
        self.sl = self._GetSaveLoadParser()
        self.image_size = 299
        
        self.SetArgs(self.tr,       image_size=self.image_size)
        self.SetArgs(self.model,    image_size=self.image_size)
        self.SetArgs(self.aug,      image_size=self.image_size)
        
    @staticmethod
    def SetArgs(cfg, **kwargs):
        for name, value in kwargs.items():
            if not hasattr(cfg, name):
                raise ValueError('term {} not defined'.format(name))
            setattr(cfg, name, value)
            
    @staticmethod
    def _GetModelParser():
        parser = argparse.ArgumentParser()
        choices_loss = ['softmax', 'center']
        choices_optimizer = ['adagrad', 'adam']
        
        info_n_classes = 'number of classes in the training data'
        info_image_size = 'network input image size'
        info_reg = 'regularization for weights of cov layers and fc layers. Biases are excluded'
        info_loss_method = 'loss methord to use. There are two options: softmax and center'
        info_alpha = 'learning rate of centers. only used when set loss to center. '
        info_alpha += 'for more details, please check the paper <A Discriminative Feature Learning Approach for Deep Face Recognition>'
        info_lambda = 'center softmax tradeoff. only used when set loss to center. '
        info_lambda += 'note that the config name is lmbda not lambda, since lambda is a fuckin python key word'
        info_optimizer = 'optimizer to use'
        
        #loss_method
        parser.add_argument('--n_classes',      type=int,       help=info_n_classes,        default=None)
        parser.add_argument('--image_size',     type=int,       help=info_image_size,       default=None)
        parser.add_argument('--reg',            type=float,     help=info_reg,              default=0.0004)
        parser.add_argument('--loss_method',    type=str,       help=info_loss_method,      default='center', choices=choices_loss)
        parser.add_argument('--alpha',          type=float,     help=info_alpha,            default=0.1)
        parser.add_argument('--lmbda',          type=float,     help=info_lambda,           default=0.01)
        parser.add_argument('--optimizer',      type=str,       help=info_optimizer,        default='adam', choices=choices_optimizer)
        
        return parser.parse_args([])
    
    
    @staticmethod
    def _GetTrainParser():
        path_root = 'F:\\FV_TMP\\Data\\Full\\color_320'
        path_txt = 'F:\\FV_TMP\\Data\\Full\\list_tr.txt'
        
        parser = argparse.ArgumentParser()
        info_path_root = 'root dir of images'
        info_path_txt = 'path of the txt file'
        info_lr = 'initial learning rate. Later, there will be a mapping system between learning rate and epoch'
        info_max_epoch = 'how many epoch to train'
        info_verbose = 'if true, print training infomation'
        info_print_every = 'print training infomation repeatedly every <value> batch'
        info_mapper = 'dynamic learning rate.'
        info_image_size = 'network input image size'
        mapper = LearningRateMapper()
        
        parser.add_argument('--path_root',      type=str,       help=info_path_root,    default=path_root)
        parser.add_argument('--path_txt',       type=str,       help=info_path_txt,     default=path_txt)
        parser.add_argument('--seed',           type=int,       help='',                default=231)
        parser.add_argument('--batch_size',     type=int,       help='',                default=32)
        parser.add_argument('--lr',             type=float,     help=info_lr,           default=0.01)
        parser.add_argument('--max_epoch',      type=int,       help=info_max_epoch,    default=30)
        parser.add_argument('--verbose',        type=bool,      help=info_verbose,      default=True)
        parser.add_argument('--print_every',    type=int,       help=info_print_every,  default=50)
        parser.add_argument('--lr_mapper', type=LearningRateMapper, help=info_mapper,   default=mapper)
        parser.add_argument('--image_size',     type=int,       help=info_image_size,   default=None)

        return parser.parse_args([])
        
    @staticmethod
    def _GetValidationParser():
        lfw_dir = 'F:\\FV_TMP\\Data\\Full\\Aligned_160'
        lfw_pairs = 'F:\\FV_TMP\\Data\\Center\\pairs_small.txt'
        
        path_root = 'F:\\FV_TMP\\Data\\Full\\color_299'
        path_txt = 'F:\\FV_TMP\\Data\\Full\\list_val.txt'
        parser = argparse.ArgumentParser()
        
        # LFW configuration (used to evaluate AUC of face varification on LFW dataset)
        info_use_lfw = 'if true, use LFW dataset to evaluate AUC performance of face varification task'
        info_lfw_dir = 'directory of LFW images'
        info_lfw_pairs = 'path of pairs.txt'
        info_lfw_every = 'validate every <value> epochs'
        parser.add_argument('--use_lfw',        type=bool,      help=info_use_lfw,          default=False)
        parser.add_argument('--lfw_dir',        type=str,       help=info_lfw_dir,          default=lfw_dir)
        parser.add_argument('--lfw_pairs',      type=str,       help=info_lfw_pairs,        default=lfw_pairs)
        parser.add_argument('--lfw_every',      type=int,       help=info_lfw_every,        default=1)
        
        # validation configuration (used to evaluate accuracy of softmax classification on testing dataset)
        info_use_val = 'if true, use validation dataset to evaluate classification performance'
        info_path_root = 'root dir of images'
        info_path_txt = 'path of the txt file'
        info_val_every = 'validate every <value> epochs'
        parser.add_argument('--use_val',        type=bool,      help=info_use_val,          default=True)
        parser.add_argument('--path_root',      type=str,       help=info_path_root,        default=path_root)
        parser.add_argument('--path_txt',       type=str,       help=info_path_txt,         default=path_txt)
        parser.add_argument('--val_every',      type=int,       help=info_val_every,        default=1)
        return parser.parse_args([])
        
    @staticmethod
    def _GetAugmentationParser():
        parser = argparse.ArgumentParser()
    
        info_image_size = 'network input image size'
        info_random_crop = 'Performs random cropping of training images. '
        info_random_crop += 'If false, the center image_size pixels from the training images are used. '
        info_random_crop += 'If the size of the images in the data directory is equal to image_size no cropping is performed'
        info_random_flip = 'Performs random horizontal flipping of training images'
        info_random_rotate = 'Performs random rotations of training images'
        info_normalize = 'Performs nomalization on image'
        
        parser.add_argument('--image_size',         type=int,   help=info_image_size,       default=None)
        parser.add_argument('--random_crop',        type=bool,  help=info_random_crop,      default=True)
        parser.add_argument('--random_flip',        type=bool,  help=info_random_flip,      default=True)
        parser.add_argument('--random_rotate',      type=bool,  help=info_random_rotate,    default=True)
        parser.add_argument('--normalize',          type=bool,  help=info_normalize,        default=True)
        return parser.parse_args([])
    
        
    @staticmethod
    def _GetSaveLoadParser():
        save_dir = 'F:\\FV_TMP\\model\\Res_Inception_V2'
        load_dir = 'F:\\FV_TMP\\model'
        parser = argparse.ArgumentParser()
       
        info_is_save = 'if true, save the session to checkpoint file after <value> iteration'
        info_save_every = 'save the session to disk every <value> epoch '
        info_save_dir = 'save model to directory <value>'
        info_is_load = 'if true, initialize the session from checkpoint file'
        info_load_dir = 'load checkpoint from directory <value> '
        
        parser.add_argument('--is_save',        type=bool,      help=info_is_save,      default=True)
        parser.add_argument('--save_every',     type=int,       help=info_save_every,   default=2)
        parser.add_argument('--save_dir',       type=str,       help=info_save_dir,     default=save_dir)
        parser.add_argument('--is_load',        type=bool,      help=info_is_load,      default=True)
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
        #self._Add(iteration=45,   decay_factor=0.1)
        
    def _Compute(self):
        self.data = pd.DataFrame([])
        self.data['iter'] = pd.Series(self._iteration)
        self.data['decay'] = pd.Series(self._decay)
        self.data.sort_values(by='iter', ascending=True, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
    def GetFactor(self, iteration):
        which = np.sum(self.data['iter'] <= iteration) - 1
        return self.data.loc[which, 'decay']