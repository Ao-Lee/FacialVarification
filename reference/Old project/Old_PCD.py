'''
该文件用于测试业务数据集，并将业务数据集的数据(一对img看成一份数据)分成三类
分别是'same person', 'different person'和'unable to locate face'
本程序建立对应这三个文件夹，并将每条数据(一对img)分类到其中一个文件夹中


业务数据集的规范如下：
    (1)业务数据集是一个文件夹
    (2)业务数据集文件夹下包含多个子文件夹，名称不限
    (3)每个子文件夹包含两张图片（不能多，不能少，只能是两张）
    (4)每张图片的后缀为'.jpg'或'.jpeg'
    (5)业务数据集文件夹的默认路径是'data/业务'
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import fv.AlignUtils as AlignUtils
import fv.ModelUtils as ModelUtils
import fv.DataUtils as DataUtils
import os
import pandas as pd
import shutil

def GetPCDDataFrame(dir_parant):
    def _GetAllFiles(path):
        'we assume that there are only two img files under path directory'
        directories = []
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            cond1 = os.path.isfile(full_path)
            cond2 = full_path.find('.jpeg') != (-1) or full_path.find('.jpg') != (-1)
            if cond1 and cond2:
                directories.append(full_path)
        if len(directories)!=2:
            return None
        else:
            return directories
            
    list_name = []
    list_path0 = []
    list_path1 = []
    for dir_sub in os.listdir(dir_parant):
        dir_full = os.path.join(dir_parant, dir_sub)
        if not os.path.isdir(dir_full):
            continue
        directories = _GetAllFiles(dir_full)
        assert len(directories)==2
        if directories is None:
            continue
        list_name.append(dir_sub)
        list_path0.append(directories[0])
        list_path1.append(directories[1])
        
    #construct dataframe from list_name, list_path0, and list_path1
    df = pd.DataFrame([])
    df['name1'] = pd.Series(list_name)
    df['name2'] = pd.Series(list_name)
    df['path1'] = pd.Series(list_path0)
    df['path2'] = pd.Series(list_path1)
    return df
    
def GetResult(path):
    df = GetPCDDataFrame(path)
    Align = AlignUtils.GetAlignFunc()
    prediction = []
    total_iteration = len(df)
    for idx in range(total_iteration):
        if idx%10 == 0:
            print('current iteration is {}/{}'.format(idx,total_iteration))
        path1 = df.loc[idx,'path1']
        path2 = df.loc[idx,'path2']
        img1 = Align(path1)
        img2 = Align(path2)
        if img1 is None or img2 is None:
            prediction.append('unable to locate face')
            continue
        img1, _, _ = DataUtils.Whiten(img1)
        img2, _, _ = DataUtils.Whiten(img2)
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        pair = np.concatenate([img1,img2], axis=0)
        embeddings = ModelUtils.GetEmbeddings(pair)
        dist = ((embeddings[0] - embeddings[1])**2).sum()
        if dist > ModelUtils.GetThreshold():
            prediction.append('different person')
        else:
            prediction.append('same person')
    # construct result data frame
    result = pd.DataFrame([])
    result['name1'] = df['name1']
    result['name2'] = df['name2']
    result['result'] = pd.Series(prediction)
    return result
    
def CopyFiles(df, path_from, path_to):
    categories = ['different person', 'same person', 'unable to locate face']
    for category in categories:
        print('creating file {}'.format(path_to + os.sep + category))
        mask = df['result'] == category
        df_selected = df[mask]
        df_selected.reset_index(drop=True,inplace=True)
        
        for idx in range(len(df_selected)):
            old_dir_full = path_from + os.sep + df_selected.loc[idx, 'name1']
            new_dir_full = path_to + os.sep + category + os.sep + df.loc[idx, 'name1']
            shutil.copytree(old_dir_full, new_dir_full)
       
def GenerateResultFolder(path_from, path_to):
    df = GetResult(path_from)
    CopyFiles(df, path_from, path_to)
    grouped = df['result'].groupby(df['result'])
    print(grouped.count())
    
def ShowFolderImg(folder_path, title, align_func):
    path_list = []
    for item in os.listdir(folder_path):
        current_item = os.path.join(folder_path, item)
        if os.path.isfile(current_item):
            path_list.append(current_item)
    assert len(path_list) == 2
    img0 = align_func(path_list[0])
    img1 = align_func(path_list[1])
    if img0 is None or img1 is None:
        return
    DataUtils.ShowPair(img0, img1, title)
    
def ShowPicture(path_to):
    categories = ['same person', 'different person']
    align_func = AlignUtils.GetAlignFunc()
    for category in categories:
        path_category = path_to + os.sep + category
        iteration = 0
        for item in os.listdir(path_category):
            path_current = os.path.join(path_category, item)
            ShowFolderImg(path_current, category, align_func)
            iteration += 1
            if iteration > 15:
                break
            
if __name__ == '__main__':
    path_from = 'F:\\Dropbox\\PCD\\data\\业务\\亿图预测为相同人'
    path_to = 'F:\\Dropbox\\PCD\\data\\业务\\result'
    path_model = 'F:\\Dropbox\\PCD\\src\\models'
    ModelUtils.InitSessionAndGraph(path_model)
    
    GenerateResultFolder(path_from, path_to)
    ShowPicture(path_to)
    
