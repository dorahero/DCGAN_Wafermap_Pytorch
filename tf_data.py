#%%
from load_for_tf import load_wm_811k
import pandas as pd
import cv2
from tqdm import tqdm

df = pd.read_pickle("./data/LSWMD.pkl/LSWMD.pkl")

data = load_wm_811k(df, 64)
print(data)

#%%

for i, img in enumerate(data['waferMap']):
    thresh = img*255
    cv2.imwrite(f"data/near_full_training/{i}.png", thresh)


# %%
