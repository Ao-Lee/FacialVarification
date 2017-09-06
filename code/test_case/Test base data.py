import numpy as np
from fv.UtilsData import DataLoader, Dataset


class TestData(Dataset):

    def __init__(self):
        self.length = 200
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        label = np.random.randint(0,2)
        img1 = np.random.random([72,72,3])
        img2 = np.random.random([72,72,3])
        dictionary = {'img1':img1, 'img2':img2}
        return dictionary, label

def GetDataLoader(batch_size=12):
    mydataset = TestData()
    mydataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=True)
    return mydataloader
    
if __name__=='__main__':
    loader = GetDataLoader()
    for iteration,(dictionary,label) in enumerate(loader):
        print(label.shape)
        print(dictionary['img1'].shape)
        print(dictionary['img2'].shape)
        break
