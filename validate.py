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
    
    
# validate accuracy performance on classification task
def ValidateACC(model, session, dataloader, is_print=True):
    acc = AverageMeter()
    softmax = AverageMeter()
    center = AverageMeter()
    reg = AverageMeter()
    total = AverageMeter()
    for imgs, _, labels in dataloader:
        size = imgs.shape[0]
        feed = {model.pl_images:imgs, model.pl_labels:labels, model.pl_phase:True}
        target = [model.accuracy, model.loss_softmax, model.loss_center, model.loss_reg]
        v_acc, v_soft, v_center, v_reg = session.run(target, feed_dict=feed)
        acc.update(v_acc, size)
        softmax.update(v_soft, size)
        center.update(v_center, size)
        reg.update(v_reg, size)
        total.update(v_soft + v_center + v_reg, size)
    
    if is_print:
        print('---------- validation on testing data ----------')
        info = 'acc:{0:6.3f}  softmax:{1:6.3f}  center:{2:6.3f}  reg:{3:6.3f}  total:{4:6.3f}'
        print(info.format(acc.avg, softmax.avg, center.avg, reg.avg, total.avg))
        print('\n\n')
        
    return acc.avg
        
if __name__=='__main__':
    pass