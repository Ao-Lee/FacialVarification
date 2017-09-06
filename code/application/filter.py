from fv.data import LFWSet
import shutil
from os import listdir
from os.path import isdir, join

def _GetBanNames(dir_pairs,dir_source):
    lfw = LFWSet(dir_pairs, dir_source, suffix='jpg')
    s = set(lfw.df['name1']) | set(lfw.df['name2'])
    return s

def _GetFullNames(dir_source):
    dirs = [d for d in listdir(dir_source) if isdir(join(dir_source, d))]
    return set(dirs)

def _GetValidateNames(dir_pairs,dir_source):
    ban = _GetBanNames(dir_pairs,dir_source)
    full = _GetFullNames(dir_source)
    pick = full - ban
    assert len(pick) + len(ban) == len(full)
    return list(pick)

'''
把LFW数据集复制到另外一个目录，并过滤掉在pairs.txt中出现人名的文件夹
dir_pairs是pairs.txt的路径
dir_source是源LFW数据集路径
dir_target是目标LFW数据集路径
'''
def FilterLFW(dir_pairs,dir_source,dir_target):
    names = _GetValidateNames(dir_pairs,dir_source)
    for name in names:
        current_from = join(dir_source, name)
        current_to = join(dir_target, name)
        shutil.copytree(current_from, current_to)

def TestFilter():
    dir_source = 'F:\\FV_TMP\\Data\\Aligned_320'
    dir_target = 'F:\\FV_TMP\\Data\\Filtered_Small_320'
    dir_pairs = 'F:\\FV_TMP\\Data\\pairs_small.txt'
    FilterLFW(dir_pairs,dir_source,dir_target)
    
if __name__=='__main__':
    TestFilter()
    
 