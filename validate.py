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
    
def _TrainOneEpoch(model, session, dataloader, cfg, epoch):
    lr = cfg.tr.lr * cfg.tr.lr_mapper.GetFactor(epoch)
    total_batch = len(dataloader)
    a_softmax = AverageMeter()
    a_center = AverageMeter()
    a_reg = AverageMeter()
    a_total = AverageMeter()
    a_acc = AverageMeter()

    # record current epoch number to the model, so that
    # when the model was loaded from a checkpoint file,
    # the model would know how many epoches it had runned
    session.run(model.op_assign_epoch, feed_dict={model.pl_epoch:epoch})
    
    for batch, (imgs, _, labels) in enumerate(dataloader):
        feed = {model.pl_images:imgs, model.pl_labels:labels, model.pl_phase:True, model.pl_lr:lr}
        target = [model.accuracy, model.loss_softmax, model.loss_center, model.loss_reg, model.op_train]
        result = session.run(target, feed_dict=feed)

        acc, l_softmax, l_center, l_reg, _ = result
        a_softmax.update(l_softmax, imgs.shape[0])
        a_center.update(l_center, imgs.shape[0])
        a_reg.update(l_reg, imgs.shape[0])
        total = l_softmax + l_center + l_reg
        a_total.update(total, imgs.shape[0])
        a_acc.update(acc, imgs.shape[0])
        
        if cfg.tr.verbose and batch % cfg.tr.print_every == 0:
            info = '[{0}][{1:3}/{2}]  acc:{3:6.3f}  softmax:{4:6.3f}  center:{5:6.3f}  reg:{6:6.3f}  total:{7:6.3f}'
            print(info.format(epoch, batch, total_batch, acc, l_softmax, l_center, l_reg, total))
    print('\n\n')
            
    auc = None
    if cfg.val.use_lfw and epoch % cfg.val.lfw_every == 0:
        lfw = LFWSet(cfg.val.lfw_pairs, cfg.val.lfw_dir, cfg.aug)
        dl = DataLoader(lfw, batch_size=16, shuffle=False)
        auc = ValidateAUC(model, session, dl, is_print=cfg.tr.verbose)
    
    acc_val = None
    if cfg.val.use_val and epoch % cfg.val.val_every == 0:
        data_val = TxtSet(cfg.val.path_root, cfg.val.path_txt, cfg.aug, processing='validation')
        dl = DataLoader(data_val, batch_size=16, shuffle=False)
        acc_val = ValidateACC(model, session, dl, is_print=cfg.tr.verbose)
        
    if cfg.sl.is_save and epoch % cfg.sl.save_every == 0:
        _SaveModel(session, model, cfg.sl.save_dir, epoch, is_print=cfg.tr.verbose)

    epoch_result = {}
    epoch_result['epoch'] = epoch
    epoch_result['softmax'] = a_softmax.avg
    epoch_result['reg'] = a_reg.avg
    epoch_result['center'] = a_center.avg
    epoch_result['total_loss'] = a_total.avg
    epoch_result['accuracy'] = a_acc.avg
    if auc is not None:
        epoch_result['auc'] = auc
    if acc_val is not None:
        epoch_result['acc_val'] = acc_val
    
    if cfg.tr.verbose:
        print('---------- average statistics for the whole epoch ----------')
        info = 'acc:{0:6.3f}  softmax:{1:6.3f}  center:{2:6.3f}  reg:{3:6.3f}  total:{4:6.3f}'
        print(info.format(epoch_result['accuracy'], epoch_result['softmax'], epoch_result['center'], epoch_result['reg'], epoch_result['total_loss']))
        print('\n\n')
        
    return epoch_result
        
def _Run(model, session, dataloader, cfg):
    
    current_epoch = session.run(model.epoch) + 1
    print('the training process starts from epoch {}'.format(current_epoch))
    results = []
    for epoch in range(current_epoch, cfg.tr.max_epoch):
       result = _TrainOneEpoch(model, session, dataloader, cfg, epoch)
       
       if result is not None:
           results.append(result)
       
    collected_results = {key: [result[key] for result in results] for key in results[0]}
    df = pd.DataFrame(collected_results)
    #columns = ['epoch', 'auc', 'accuracy', 'total_loss', 'softmax', 'center', 'reg']
    #df = df[columns]
    return df
    
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