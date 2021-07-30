from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import pandas as pd 


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

    # df.to_csv("data/LSWM.csv")

    # tol_wafers = df.shape[0]

    # df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
    # df_withlabel = df_withlabel.reset_index()
    df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']==7)]
    df_withpattern = df_withpattern.reset_index()
    # df_nonpattern = df[(df['failureNum']==8)]

    data = pd.DataFrame()
    labelencoder = LabelEncoder()
    data['waferMap'] = df_withpattern['waferMap'].apply(lambda x: cv2.resize(np.clip(np.array(x), 1, 2) - 1, (img_size, img_size), interpolation = cv2.INTER_NEAREST))
    data['labels'] = labelencoder.fit_transform(df_withpattern['failureType'].apply(lambda x: x[0][0]))

    return data