from fv import TrainingSet, TrainModel
from fv.UtilsData import DataLoader
from fv.UtilsConfig import Cfg

if __name__ == '__main__':
    cfg = Cfg()
    training_set = TrainingSet(cfg.train.data_dir, cfg.augmentation)
    dataloader = DataLoader(training_set, batch_size=cfg.train.batch_size, shuffle=True)
    cfg.model.n_classes = len(training_set.mapping)
    info = TrainModel(cfg, dataloader)