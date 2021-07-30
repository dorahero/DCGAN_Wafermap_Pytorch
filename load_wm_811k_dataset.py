import pandas as pd 
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import torch

class CustomWafermapDataset(Dataset):
    def __init__(self, image, labels, img_size):
        self.labels = labels
        self.image = image
        self.img_size = img_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.image[idx]
        data = np.asarray(data).astype(np.uint8).reshape(1, self.img_size, self.img_size)
        sample = (data, label)
        return sample


def load_wm_811k(df, img_size):
    df = df.drop(['waferIndex'], axis = 1)

    def find_dim(x):
        dim0 = np.size(x,axis=0)
        dim1 = np.size(x,axis=1)
        return dim0,dim1
    df['waferMapDim'] = df.waferMap.apply(find_dim)


    df['failureNum'] = df.failureType
    df['trainTestNum'] = df.trianTestLabel
    mapping_type = {'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
    mapping_traintest = {'Training':0,'Test':1}
    df = df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

    tol_wafers = df.shape[0]

    df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
    df_withlabel = df_withlabel.reset_index()
    df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']==0)]
    df_withpattern = df_withpattern.reset_index()
    df_nonpattern = df[(df['failureNum']==8)]

    data = pd.DataFrame()
    labelencoder = LabelEncoder()
    data['waferMap'] = df_withpattern['waferMap'].apply(lambda x: cv2.resize(np.clip(np.array(x), 1, 2) - 1, (img_size, img_size), interpolation = cv2.INTER_NEAREST))
    data['labels'] = labelencoder.fit_transform(df_withpattern['failureType'].apply(lambda x: x[0][0]))

    W_Dataset = CustomWafermapDataset(data['waferMap'], data['labels'], img_size)
    
    return W_Dataset

def load_new_scratch(path, img_size):
    import glob
    import cv2
    from tqdm import tqdm
    imgs = glob.glob(path+"/*.png")
    
    waferMap = []
    print("read images")
    for img in tqdm(imgs):
        image = cv2.imread(img, 0)
        image = cv2.resize(image, (img_size, img_size), interpolation = cv2.INTER_NEAREST)
        waferMap.append(image/255.)

    labels = [0 for i in range(len(waferMap))]

    W_Dataset = CustomWafermapDataset(waferMap, labels, img_size)
    
    return W_Dataset