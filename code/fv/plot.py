import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()
    plt.close()
    
# batch     (B, 160, 160, 3)
# titles    a list of strings of size B
def ShowBatch(batch, titles):
    B = batch.shape[0]
    assert B == len(titles)
    for idx in range(B):
        plt.figure()
        plt.imshow(batch[idx].astype('uint8'))
        plt.title(titles[idx])
        plt.show()
        plt.close()
    
    
    
    
    