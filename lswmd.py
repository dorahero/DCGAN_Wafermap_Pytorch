#%%
import pandas as pd 
df = pd.read_pickle("../data/LSWMD.pkl/LSWMD.pkl")

#%%
from load_wm_811k_dataset import load_wm_811k
import torch

W_Dataset = load_wm_811k(df, 64)

dataloader = torch.utils.data.DataLoader(
    W_Dataset,
    batch_size=128,
    shuffle=True,
)

#%%
Tensor = torch.cuda.FloatTensor
from torch.autograd import Variable

import cv2
import numpy as np
for i, (imgs, _) in enumerate(dataloader):
    real_imgs = Variable(imgs.type(Tensor))
    for j, img in enumerate(np.array(real_imgs.cpu()).astype(float)):
        thresh = img*255
        # print(thresh[0][0])
        cv2.imwrite(f"png/tmp_{i}_{j}.png", thresh[0])
        

#%%
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

import torch

class CustomWafermapDataset(Dataset):
    def __init__(self, image, labels):
        self.labels = labels
        self.image = image

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.image[idx]
        data = np.asarray(data).astype(np.uint8).reshape(1, 64, 64)
        sample = (data, label)
        return sample


 # %%

dataloader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=128,
    shuffle=True,
)

for m in dataloader_mnist:
    print(m[0])
    break
#%%
from load_wm_811k_dataset import load_wm_811k

W_Dataset = load_wm_811k(df)

dataloader = torch.utils.data.DataLoader(
    W_Dataset,
    batch_size=128,
    shuffle=True,
)
for m in dataloader:
    print(m[0])
    break

# %%
df = df.drop(['waferIndex'], axis = 1)

# %%
def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1
df['waferMapDim']=df.waferMap.apply(find_dim)
df.sample(5)
# %%
max(df.waferMapDim), min(df.waferMapDim)

# %%
uni_waferDim=np.unique(df.waferMapDim, return_counts=True)
uni_waferDim[0].shape[0]
# %%
df['failureNum']=df.failureType
df['trainTestNum']=df.trianTestLabel
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
mapping_traintest={'Training':0,'Test':1}
df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})
# %%
tol_wafers = df.shape[0]
tol_wafers
# %%
df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
df_withlabel =df_withlabel.reset_index()
df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
df_withpattern = df_withpattern.reset_index()
df_nonpattern = df[(df['failureNum']==8)]
df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]
# %%
import matplotlib.pyplot as plt
%matplotlib inline

from matplotlib import gridspec
fig = plt.figure(figsize=(20, 4.5)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5]) 
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

no_wafers=[tol_wafers-df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]]

colors = ['silver', 'orange', 'gold']
explode = (0.1, 0, 0)  # explode 1st slice
labels = ['no-label','label&pattern','label&non-pattern']
ax1.pie(no_wafers, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

uni_pattern=np.unique(df_withpattern.failureNum, return_counts=True)
labels2 = ['','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
ax2.bar(uni_pattern[0],uni_pattern[1]/df_withpattern.shape[0], color='gold', align='center', alpha=0.9)
ax2.set_title("failure type frequency")
ax2.set_ylabel("% of pattern wafers")
ax2.set_xticklabels(labels2)

plt.show()
# %%
fig, ax = plt.subplots(nrows = 10, ncols = 10, figsize=(20, 20))
ax = ax.ravel(order='C')
for i in range(100):
    img = df_withpattern.waferMap[i]
    ax[i].imshow(img)
    ax[i].set_title(df_withpattern.failureType[i][0][0], fontsize=10)
    ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show() 
# %%
x = [0,1,2,3,4,5,6,7]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

for k in x:
    fig, ax = plt.subplots(nrows = 1, ncols = 10, figsize=(18, 12))
    ax = ax.ravel(order='C')
    for j in [k]:
        img = df_withpattern.waferMap[df_withpattern.failureType==labels2[j]]
        for i in range(10):
            ax[i].imshow(img[img.index[i]])
            ax[i].set_title(df_withpattern.failureType[img.index[i]][0][0], fontsize=10)
            ax[i].set_xlabel(df_withpattern.index[img.index[i]], fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    plt.tight_layout()
    plt.show() 
# %%
x = [9,340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

#ind_def = {'Center': 9, 'Donut': 340, 'Edge-Loc': 3, 'Edge-Ring': 16, 'Loc': 0, 'Random': 25,  'Scratch': 84, 'Near-full': 37}
fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    img = df_withpattern.waferMap[x[i]]
    ax[i].imshow(img)
    ax[i].set_title(df_withpattern.failureType[x[i]][0][0],fontsize=24)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show() 
# %%
df_withpattern
#%%
import cv2
import numpy as np
data = pd.DataFrame()
labelencoder = LabelEncoder()
data['waferMap'] = df_withpattern['waferMap'].apply(lambda x: cv2.resize(np.array(x), (64, 64), interpolation = cv2.INTER_NEAREST))
data['labels'] = labelencoder.fit_transform(df_withpattern['failureType'].apply(lambda x: x[0][0]))

#%%
data

# %%
W_Dataset = CustomWafermapDataset(data['waferMap'], data['labels'])


dataloader = torch.utils.data.DataLoader(
    W_Dataset,
    batch_size=128,
    shuffle=True,
)

# %%
for m in dataloader:
    print(m[0])
    break
# %%
