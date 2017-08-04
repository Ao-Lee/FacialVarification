import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

try:
    from avg import AverageMeter
except ImportError:
    from .avg import AverageMeter
    
'''
emb1:   ndarray with size (batch, D)
emb2:   ndarray with size (batch, D)
return: list with length (batch)
'''
def _GetCosineDistance(feature1, feature2):
    batch = feature1.shape[0]
    return [cosine(feature1[idx,:], feature2[idx,:]) for idx in range(batch)]
    
'''
emb1:   ndarray with size (batch, D)
emb2:   ndarray with size (batch, D)
return: list with length (batch)
'''
def _GetEuclideanDistance(emb1, emb2):
    diff = (emb1 - emb2)
    dist = (diff**2).sum(axis=1)
    return list(dist)

def _PlotROC(fpr, tpr, title=''):
    plt.plot(fpr, tpr)
    plt.title(title)
    plt.xlabel("F P R")
    plt.ylabel("T P R")
    plt.show()
    plt.close()
    
# validate auc performance on face varification task
def ValidateAUC(model, session, dataloader, is_print=True):
    
    dists_euclidean = []
    dists_cosine = []
    labels = []
    
    for data, _, label in dataloader:
        imgs1 = data['img1']
        imgs2 = data['img2']
        feed1 = {model.pl_images:imgs1, model.pl_phase:False}
        feed2 = {model.pl_images:imgs2, model.pl_phase:False}
        emb1, f1 = session.run([model.embeddings, model.deep_features], feed_dict=feed1)
        emb2, f2 = session.run([model.embeddings, model.deep_features], feed_dict=feed2)
        dists_euclidean += _GetEuclideanDistance(emb1, emb2)
        dists_cosine += _GetCosineDistance(f1, f2)
        labels += list(label)

    dists_euclidean = np.array(dists_euclidean)
    dists_cosine = np.array(dists_cosine)
    labels = np.array(labels)
    
    labels = np.logical_not(np.array(labels))
    auc = metrics.roc_auc_score(labels, dists_euclidean)

    if is_print:
        print('---------- validation on LFW ----------')
        fpr, tpr, thresholds1 = metrics.roc_curve(labels, dists_euclidean)
        _PlotROC(fpr, tpr, title='Euclidean Distance ROC')
        print('auc is:\t{}'.format(auc))
        print('\n\n')
    return auc
    
    
def TrainModel(cfg, dataloader):
    g = tf.Graph()
    with g.as_default():
        np.random.seed(seed=cfg.tr.seed)
        random.seed(cfg.tr.seed)
        tf.set_random_seed(cfg.tr.seed)
        model = Inception_Res_V1(cfg=cfg.model)
        session_cfg =tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        session = tf.Session(config=session_cfg)
  
        if cfg.sl.is_load:
            print('begin to load model from file')
            ckpt = tf.train.get_checkpoint_state(cfg.sl.load_dir)
            assert ckpt and ckpt.model_checkpoint_path, 'unable to load checkpoint'
            model.saver.restore(session, ckpt.model_checkpoint_path)
            print('model sucessfully loaded')
        else:
            print('begin to generate a new model, the model will be trained from scratch')
            session.run(model.op_init)
            print('model sucessfully created')
            
        info = _Run(model, session, dataloader, cfg)
        session.close()
    return info
        
if __name__=='__main__':
    pass