import numpy as np
import matplotlib.pyplot as plt

# x         (160, 160, 3)
# return    (160, 160, 3)
def Whiten(img):
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0/np.sqrt(img.size))
    whiten = np.multiply(np.subtract(img, mean), 1/std_adj)
    return whiten, mean, std_adj
 
# x         (Batch, 160, 160, 3)
# mean      (Batch,)
# std       (Batch,)
# return    (Batch, 160, 160, 3)
def ReverseWhiten(x, mean, std):
    mean = mean.reshape(-1, 1, 1, 1)
    std = std.reshape(-1, 1, 1, 1)
    result = np.add(np.multiply(x, std), mean)
    return result.astype('uint8')

# img1    (160, 160, 3)  ndarray
# img2    (160, 160, 3)  ndarray
def ShowPair(img1, img2, title, padding=20):
    assert img1.shape == img2.shape
    h, w, c = img1.shape
    p = padding
    result_size = [h+p*2 , w*2+p*3 , c]
    result = np.zeros(result_size)
    result[p:p+h , p:p+w , :] = img1
    result[p:p+h , 2*p+w:2*p+2*w , :] = img2
    plt.figure()
    plt.imshow(result.astype('uint8'))
    plt.title(title)
    
if __name__=='__main__':
    pass

    