from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__=='__main__':
    path = 'F:\\Dropbox\\DataScience\\FacialVarificationProject\\data\\MNIST_PYTORCH'
    trans = transforms.Compose(
                    [
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                    ])
    
    dataset_tr = MNIST(root=path, train=True, download=False, transform=trans)
    dataset_te = MNIST(root=path, train=False, download=False, transform=trans)
    
    train_loader = DataLoader(dataset_tr)
    test_loader = DataLoader(dataset_te)
    
    for img, label in train_loader:
        print(img.shape)
        print(label.shape)
        break
    