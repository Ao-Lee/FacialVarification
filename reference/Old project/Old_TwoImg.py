'''
比较两张图片，返回结果
'''
import os
import fv.AlignUtils as AlignUtils
import fv.ModelUtils as ModelUtils
import fv.DataUtils as DataUtils
import numpy as np

def Compare(path1, path2, align_func):
    img1 = align_func(path1)
    img2 = align_func(path2)
    if img1 is None or img2 is None:
        return 'unable to locate face'
    img1, _, _ = DataUtils.Whiten(img1)
    img2, _, _ = DataUtils.Whiten(img2)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    pair = np.concatenate([img1,img2], axis=0)
    embeddings = ModelUtils.GetEmbeddings(pair)
    dist = ((embeddings[0] - embeddings[1])**2).sum()
    print(dist)
    print(ModelUtils.GetThreshold())
    if dist > ModelUtils.GetThreshold():
        return 'different person'
    else:
        return 'same person'
        
if __name__ == '__main__':
    path_data = 'F:\\Dropbox\\PCD\\data\\TwoImg'
    path_model = 'F:\\Dropbox\\PCD\\src\\models'

    directories = []
    for item in os.listdir(path_data):
        full_path = os.path.join(path_data, item)
        cond1 = os.path.isfile(full_path)
        cond2 = full_path.find('.jpeg') != (-1) or full_path.find('.jpg') != (-1)
        if cond1 and cond2:
            directories.append(full_path)
    assert len(directories)==2

    ModelUtils.InitSessionAndGraph(path_model)
    align_func = AlignUtils.GetAlignFunc()
    result = Compare(directories[0], directories[1], align_func)
    print(result)

    