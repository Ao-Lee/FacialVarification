from fv.data import FolderSet
from fv.UtilsData import DataLoader
from fv.UtilsConfig import Cfg
import fv.plot as plot

if __name__=='__main__':
    cfg = Cfg()
    cfg.SetArgs(cfg.aug, normalize=False)
    tr = FolderSet(cfg.tr.data_dir, cfg.aug, processing='training')
    dtr = DataLoader(tr, batch_size=4, shuffle=True)
    for imgs, _, labels in dtr:
        print(imgs.shape)
        print(labels.shape)
        names = [tr.Label2Name(label) for label in labels]
        plot.ShowBatch(imgs, names)
        break

    te = FolderSet(cfg.tr.data_dir, cfg.aug, processing='validation')
    dte = DataLoader(te, batch_size=4, shuffle=True)
    for imgs, _, labels in dte:
        print(imgs.shape)
        print(labels.shape)
        names = [te.Label2Name(label) for label in labels]
        plot.ShowBatch(imgs, names)
        break

    
    
    