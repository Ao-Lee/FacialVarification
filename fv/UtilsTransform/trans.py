from PIL import Image
import numpy as np
import random
from . import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
class Path2PIL(object):
    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
            
class PIL2Array(object):
    def __call__(self, PILImg):
        return np.array(PILImg)
    
class Normalize(object):
    '''
    input: numpy array with shape (w, h, channel)
    This class computes (x - mean) / adjusted_stddev, 
    where mean is the average of all values in image, 
    and adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))
    stddev is the standard deviation of all values in image. 
    It is capped away from zero to protect against division by 0 
    when handling uniform images.
    '''
    
    '''
    this implementation is a numpy version of tensorflow API tf.image.per_image_standardization,
    see https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    '''
    
    def __call__(self, img):
        assert isinstance(img, (np.ndarray, np.generic)), 'input img must be a numpy array'
        assert len(img.shape) == 3, 'input img must be a 3d array, but got {}'.format(len(img.shape))                  
        num_pixels = img.shape[0] * img.shape[1]
        
        stddev = np.std(img)
        min_stddev = 1.0/np.sqrt(num_pixels)
        pixel_value_scale = np.max([stddev, min_stddev])
        pixel_value_offset = np.mean(img)
        
        return (img - pixel_value_offset)/pixel_value_scale
        
        
class RandomCrop(object):
    def __init__(self, size):
        self.size = (int(size), int(size))
        
    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):

        if not F._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
            
        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)
    
class CenterCrop(object):
    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, img):
        return F.center_crop(img, self.size)    
    
class RandomHorizontalFlip(object):
    def __call__(self, img):
        
        if not F._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
            
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img
    
class RandomRotation(object):
    def __call__(self, img):

        if not F._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
            
        angle = np.random.uniform(low=-10.0, high=10.0)
        return img.rotate(angle)
