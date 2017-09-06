'''
Note: All images should be aligend first
'''
import pandas as pd
import numpy as np
import os

try:
    from UtilsData import Dataset
    from UtilsTransform import trans
except ImportError:
    from .UtilsData import Dataset
    from .UtilsTransform import trans
    
def _GetTrainingTransformer(cfg):
    transforms = []
    transforms.append(trans.Path2PIL())
    if cfg.random_rotate:
        transforms.append(trans.RandomRotation())
    if cfg.random_crop:
        transforms.append(trans.RandomCrop(cfg.image_size))
    else:
        transforms.append(trans.CenterCrop(cfg.image_size))
    if cfg.random_flip:
        transforms.append(trans.RandomHorizontalFlip())
    transforms.append(trans.PIL2Array())
    if cfg.normalize:
        transforms.append(trans.Normalize())
    return trans.Compose(transforms)
    
def _GetValidationTransformer(cfg):
    transforms = []
    transforms.append(trans.Path2PIL())
    transforms.append(trans.CenterCrop(cfg.image_size))
    transforms.append(trans.PIL2Array())
    if cfg.normalize:
        transforms.append(trans.Normalize())
    return trans.Compose(transforms)
    

    
'''
self.df has following structure:
path            label
F:\\Pic01.jpg   0
F:\\Pic02.jpg   1
F:\\Pic04.jpg   2

initialize self.df by reading a txt file
the txt file has following structure:
Aaron_Patterson\Aaron_Patterson_0001.jpg 2
Aaron_Peirsol\Aaron_Peirsol_0002.jpg 3
Aaron_Peirsol\Aaron_Peirsol_0003.jpg 3
'''
class TxtSet(Dataset):
    def __init__(self, path_root, path_txt, augmentation_cfg, processing):
        self.df = pd.read_csv(path_txt, header=None, names=['path', 'label'], delimiter=' ')
        self.root = path_root
        self.n_classes = len(self.df['label'].drop_duplicates())
        
        if processing=='training':
            self.trans = _GetTrainingTransformer(augmentation_cfg)
        elif processing=='validation':
            self.trans = _GetValidationTransformer(augmentation_cfg)
        else:
            raise ValueError('wrong processing type')
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.df.loc[idx, 'path'])
        img = self.trans(path)
        label = self.df.loc[idx, 'label']
        return img, idx, label

'''
self.df has following structure:
path            label
F:\\Pic01.jpg   0
F:\\Pic02.jpg   1
F:\\Pic04.jpg   2

initialize self.df by directly loading a folder
'''
class FolderSet(Dataset):
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        path = self.df.loc[idx, 'path']
        img = self.trans(path)
        label = self.df.loc[idx, 'label']
        return img, idx, label
        
    def __init__(self, paths, augmentation_cfg, processing):
        lst = self._Paths2List(paths)
        self.df = self._List2DataFrame(lst)
        df_mapping = self.df[['name', 'label']].drop_duplicates().reset_index(drop=True)
        names = df_mapping['name'].values
        labels = df_mapping['label'].values
        # map class labels to person names
        self.mapping = names[labels]
        self.n_classes = len(self.mapping)
        
        # data augmentation
        if processing=='training':
            self.trans = _GetTrainingTransformer(augmentation_cfg)
        elif processing=='validation':
            self.trans = _GetValidationTransformer(augmentation_cfg)
        else:
            raise ValueError('wrong processing type')
    
    # input: 1 output: 'Aaron_Guiel'
    def Label2Name(self, label):
        return self.mapping[label]
    
    @staticmethod
    def _List2DataFrame(lst):
        df = pd.DataFrame(lst,columns=['name','path'])
        df.sort_values(by='name', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['label'] = df['name'].astype('category').cat.codes
        return df

    @staticmethod
    def _GetImgPaths(person, facedir):
        if not os.path.isdir(facedir):
            return []
        images = os.listdir(facedir)
        return [[person, os.path.join(facedir,img)] for img in images]

    '''
    input: ' F:\Data\Aligned_160  '
    output:
        [
             ['Person1','F:\\Data\\Aligned_160\\Person1\\Person1_0001.jpg'],
             ['Person1','F:\\Data\\Aligned_160\\Person1\\Person1_0002.jpg'],
             ['Person2','F:\\Data\\Aligned_160\\Person2\\Person2_0001.jpg'],
        ]
    '''
    @staticmethod
    def _Path2List(path):
        result = []
        path = os.path.expanduser(path.strip(' '))
        persons = os.listdir(path)
        for person in persons:
            facedir = os.path.join(path, person)
            result += FolderSet._GetImgPaths(person, facedir)
        return result

    # input: 'F:\\DataSet1, F:\\DataSet2, F:\\DataSet3'
    @staticmethod
    def _Paths2List(paths):
        lst = []
        for path in paths.split(','): 
            lst += FolderSet._Path2List(path)
        return lst
    
    
'''
self.df has following structure:
name1           dir1            name2       dir2
Aaron_Eckhart   F:\\Pic01.jpg   Abba_Eban   F:\\Pic02.jpg
Abba_Eban       F:\\Pic02.jpg   Abba_Eban   F:\\Pic03.jpg
Abdullah        F:\\Pic04.jpg   Abba_Eban   F:\\Pic05.jpg
'''
class LFWSet(Dataset):
    def __init__(self, dir_pairs, dir_images, augmentation_cfg):
        pairs = self._ReadPairs(dir_pairs)
        self.df = self._GenerateDataFrame(dir_images, pairs, suffix='jpg')
        self.trans = _GetValidationTransformer(augmentation_cfg)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        data = {}
        path1 = self.df.loc[idx,'dir1']
        path2 = self.df.loc[idx,'dir2']  
        data['img1'] = self.trans(path1)
        data['img2'] = self.trans(path2)
        data['name1'] = self.df.loc[idx,'name1']
        data['name2'] = self.df.loc[idx,'name2']
        label = 1 if data['name1']==data['name2'] else 0
        return data, idx, label
     
    @staticmethod
    def _GenerateDataFrame(dir_images, pairs, suffix='jpg'):
        def Name2Dir(name, which):
            return os.path.join(dir_images, name, name + '_' + '%04d' % int(which)+'.'+suffix)
        def ConvertPair(pair):
            name1 = pair[0]
            dir1 = Name2Dir(name1,pair[1])
            name2 = pair[2]
            dir2 = Name2Dir(name2,pair[3])
            return [name1, dir1, name2, dir2]
            
        pairs = [ ConvertPair(pair) for pair in pairs]
        pairs_available = [pair for pair in pairs if os.path.exists(pair[1]) and os.path.exists(pair[3])]
        
        skipped_pairs = len(pairs) - len(pairs_available)
        
        if skipped_pairs>0:
            print('Skipped %d image pairs' % skipped_pairs)
    
        df = pd.DataFrame(pairs_available, columns=['name1', 'dir1', 'name2', 'dir2'])
        return df
    
    @staticmethod
    def _ReadPairs(dir_pairs):
        pairs = []
        with open(dir_pairs, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                if len(pair)==3:
                    pair.insert(2, pair[0])
                pairs.append(pair)
        return np.array(pairs)


def _TestProcessing():
    l1 = [['liao','C:/liao1'],['tom','C:/tom1'],['ana','C:/ana1']]
    l2 = [['liao','C:/liao2'],['frank','C:/frank1'],['tom','C:/tom2']]
    data = l1 + l2
    df = pd.DataFrame(data,columns=['name','path'])
    print('----------------------------------------')
    print(df)
    print('----------------------------------------')
    df.sort_values(by='name', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df)
    print('----------------------------------------')
    df['label'] = df['name'].astype('category').cat.codes
    print(df)
    print('----------------------------------------')
    
if __name__=='__main__':
    _TestProcessing()



