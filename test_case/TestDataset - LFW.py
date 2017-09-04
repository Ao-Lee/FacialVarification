from fv.UtilsData import DataLoader
from fv.UtilsConfig import Cfg
from fv.data import LFWSet
import fv.plot as plot

if __name__=='__main__':
    dir_images = 'F:\\FV_TMP\\Data\\Center\\Aligned_160'
    dir_pairs = 'F:\\FV_TMP\\Data\\Center\\pairs_small.txt'
    
    cfg = Cfg().augmentation
    Cfg.SetArgs(cfg, normalize=False)
    lfw = LFWSet(dir_pairs, dir_images, cfg)

    dl = DataLoader(lfw, batch_size=4, shuffle=True)
    
    iteration = 0
    for data, _, label in dl:
        iteration +=1
        if iteration>=3: 
            break
        imgs1 = data['img1']
        imgs2 = data['img2']
        title1 = data['name1']
        title2 = data['name2']
        print(imgs1.shape)
        print(imgs2.shape)
        print(label.shape)
        batch_size = len(title1)
        for i in range(batch_size):
            title = title1[i] + '  <--->  ' + title2[i]
            plot.ShowPair(imgs1[i], imgs2[i], title)