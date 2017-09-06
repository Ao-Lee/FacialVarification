from fv.UtilsData import Dataset
from fv.UtilsData import DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self, length):
        self.length = length
       
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        item = np.zeros([5])
        item += idx
        return item
    
    
if __name__=='__main__':
    myset = MyDataset(22)
    dl = DataLoader(myset, batch_size=4, shuffle=True)
    for batch in dl:
        print(batch)
        
    print('************************************')
    for batch in dl:
        print(batch)
    
 

    
    
    