import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES ={
    0:'coke_bottle',
    1:'colgate_toothpaste'
}

class SKUDataset(Dataset):
    def __init__(self, paths: list, mode:str, transforms=None):
        self.paths = paths
        assert mode in ['train', 'valid', 'infer'], "Mode should be one in `train`, `valid` or `infer`"
        self.mode=mode
        self.transforms = transforms

    def __len__(self):
        print(self.paths)
        return len(list(self.paths))

    def __getitem__(self, idx):
        item ={'paths': str(self.paths[idx])}
        img = cv2.imread(item['paths'])
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        img = np.moveaxis(img, -1, 0)
        item['features']=torch.from_numpy(img)

        if self.mode != 'infer':
            item['targets'] = int(item['paths'].split('/')[-2])

        return item


        
