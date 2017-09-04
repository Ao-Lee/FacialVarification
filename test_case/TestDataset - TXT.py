from fv.data import TxtSet
from fv.UtilsData import DataLoader
from fv.UtilsConfig import Cfg
import fv.plot as plot

if __name__=='__main__':
    cfg = Cfg()
    cfg.SetArgs(cfg.aug, normalize=False)
    cfg.SetArgs(cfg.aug, image_size=128)
    path_root = 'F:\\FV_TMP\\Data\\MFM\\Aligned_128'
    path_txt = 'F:\\FV_TMP\\Data\\MFM\\list_val.txt'
    
    tr = TxtSet(path_root, path_txt, cfg.aug, processing='training')
    dtr = DataLoader(tr, batch_size=4, shuffle=True)
    for imgs, _, labels in dtr:
        print(imgs.shape)
        print(labels.shape)
        plot.ShowBatch(imgs, labels)
        break

    te = TxtSet(path_root, path_txt, cfg.aug, processing='validation')
    dte = DataLoader(te, batch_size=4, shuffle=True)
    for imgs, _, labels in dte:
        print(imgs.shape)
        print(labels.shape)
        plot.ShowBatch(imgs, labels)
        break




