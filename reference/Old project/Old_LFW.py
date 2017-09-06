'''
该文件用于测试LFW数据集
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import fv.AlignUtils as AlignUtils0
import fv.ModelUtils as ModelUtils
import fv.DataUtils as DataUtils
import sklearn.metrics as metrics
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd


class LFWSet(Dataset): # famous Labeled Faces in the Wild

    def __init__(self, dataframe, image_dir):
        self.df = dataframe
        self.image_dir = image_dir
        self.align_func = AlignUtils.GetAlignFunc()
        
    def __len__(self):
        return len(self.df)
        
    def _GetPathAndLabel(self, idx):
        name1 = self.df.loc[idx,'name1']
        name2 = self.df.loc[idx,'name2']
        pic1 = self.df.loc[idx,'pic1']
        pic2 = self.df.loc[idx,'pic2']
        label = self.df.loc[idx,'class']
        path1 = self.image_dir + os.sep + name1 + os.sep + name1 + "_%04d" % pic1 + '.jpg'
        path2 = self.image_dir + os.sep + name2 + os.sep + name2 + "_%04d" % pic2 + '.jpg'
        return path1, path2, label, name1, name2
        
    def __getitem__(self, idx):
        path1, path2, label, name1, name2 = self._GetPathAndLabel(idx)
        data = {}
        data['img1'] = self.align_func(path1)
        data['name1'] = name1
        data['img2'] = self.align_func(path2)
        data['name2'] = name2
        data['idx'] = idx
        
        return data, label
        
def GetLFWDataFrame(path_txt):
    # class == 1 if two images refer to the same person
    # class == 0 if two images refer to different persons
    df_person_different = pd.read_table(path_txt, header=None, skiprows=range(0,500))
    df_person_different.columns = ['name1','pic1','name2','pic2']
    df_person_different['class'] = 0
    df_person_same = pd.read_table(path_txt, header=None, skiprows=range(500,1000))
    df_person_same.columns = ['name1','pic1','pic2']
    df_person_same['name2'] = df_person_same['name1']
    df_person_same['class'] = 1
    df = pd.concat([df_person_same, df_person_different])
    df.reset_index(inplace=True, drop=True)
    return df

def GetLFWDataLoader(data_path, batch_size=16):
    path_txt = data_path + os.sep + 'pairsDevTest.txt'
    path_img = data_path + os.sep + 'data'
    df = GetLFWDataFrame(path_txt)
    mydataset = LFWSet(dataframe = df, image_dir=path_img)
    mydataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=True)
    return mydataloader, df
    
def GetLFWPrediction(dataloader, df):
    labels = np.array([])
    distances = np.array([])

    for iteration,(batch,label) in enumerate(dataloader):
        print('current batch is {}'.format(iteration))
        img1, _, _ = DataUtils.Whiten(batch['img1'].numpy())
        img2, _, _ = DataUtils.Whiten(batch['img2'].numpy())
        label = label.numpy()
        embedding1 = ModelUtils.GetEmbeddings(img1)
        embedding2 = ModelUtils.GetEmbeddings(img2)
        diff = (embedding1 - embedding2)
        distance = (diff**2).sum(axis=1)
        distances = np.concatenate([distances,distance])
        labels = np.concatenate([labels,label])
    return distances, labels
     
def Test01():
    '''
    show LFW data on the screen
    '''
    dataloader, _ = GetLFWDataLoader(data_path, batch_size=4)
    for iteration,(batch,label) in enumerate(dataloader):
        if iteration>=1:
            break
        imgs1 = batch['img1'].numpy()
        imgs2 = batch['img2'].numpy()
        title1 = batch['name1']
        title2 = batch['name2']
        batch_size = len(title1)
        for idx in range(batch_size):
            title = title1[idx] + '  <--->  ' + title2[idx]
            DataUtils.ShowPair(imgs1[idx], imgs2[idx], title)

def Test02():
    '''
    test performance on LFW data
    '''
    # labels == 1 means that two imgs refer to the same person
    # if distance < threshold, we predict "same person" class
    dataloader, df = GetLFWDataLoader(data_path)
    distances, labels = GetLFWPrediction(dataloader, df)
    predicted = (distances < ModelUtils.GetThreshold()).astype('int')
    confusion = metrics.confusion_matrix(labels,predicted)
    print(" Confusion matrix:\n{}".format(confusion))
    f1 = metrics.f1_score(labels,predicted)
    print(" F1 score is: {:.2f}".format(f1))
    acc = metrics.accuracy_score(labels,predicted)
    print(" accurancy is: {:.2f}".format(acc)) 
    return predicted, labels

if __name__ == '__main__':
    model_path = 'F:\\Dropbox\\PCD\\src\\models'
    data_path = 'F:\\Dropbox\\PCD\\data\\LFW'
    ModelUtils.InitSessionAndGraph(model_path=model_path)
    #Test01()
    Test02()