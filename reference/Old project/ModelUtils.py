import os
import re
import tensorflow as tf

def _GetModelFilenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
    
def _GetModel(model_path):
    model_exp = os.path.expanduser(model_path)
    print('Model directory: %s' % model_exp)
    meta_file, ckpt_file = _GetModelFilenames(model_exp)   
    print('Metagraph file: %s' % meta_file)
    print('Checkpoint file: %s' % ckpt_file)
    saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
    
def InitSessionAndGraph(model_path='models'):
    if tf.get_default_session() is None:
        tf.InteractiveSession()
        _GetModel(model_path)

def GetEmbeddings(aligned_images):
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    feed_dict = {images_placeholder:aligned_images, phase_train_placeholder:False }
    session = tf.get_default_session()
    embeddings = session.run(embeddings, feed_dict=feed_dict)
    return embeddings
    
def GetThreshold():
    threshold = 1.242
    return threshold
    
if __name__=='__main__':
    pass