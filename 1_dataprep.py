#!/usr/bin/env python
# coding: utf-8



import os
import shutil
import time
import tarfile
import json
import getpass
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from radiant_mlhub import Dataset
from skimage import io
import pylab as plt


# ## Download the data

# In[4]:


path = 'agri_data/'
# path = 'gdrive/MyDrive/agrifield_data/'


# In[5]:


if 'gdrive' in path:
    from google.colab import drive
    drive.mount('/content/gdrive')


# In[6]:


collection_name = 'ref_agrifieldnet_competition_v1'
BAND_NAMES = ['B01', 'B02', 'B03', 'B04','B05', 'B06', 'B07', 
              'B08','B8A', 'B09', 'B11', 'B12']
data_dir = path + 'data'

source_collection = f'{collection_name}_source'
train_label_collection = f'{collection_name}_labels_train'
test_label_collection = f'{collection_name}_labels_test'


# In[7]:


if not os.path.exists(data_dir):
    os.environ['MLHUB_API_KEY'] =  getpass.getpass(prompt="MLHub API Key: ")

    dataset = Dataset.fetch(collection_name)
    dataset.download(output_dir=data_dir)
    for fn in os.listdir(data_dir):
        with tarfile.open(os.path.join(data_dir, fn)) as f:
            f.extractall(data_dir + '/') 
        os.remove(os.path.join(data_dir, fn))
else:
    print("Dataset already exists")


# ## Create dataframe with list of images

# In[8]:


train_paths = os.listdir(os.path.join(data_dir, train_label_collection))
train_ids = [fn.split('_')[-1] for fn in train_paths if 'labels_train' in fn]

field_paths = [f'{data_dir}/{train_label_collection}/{train_label_collection}_{i}/field_ids.tif' 
               for i in train_ids]
label_paths = [f'{data_dir}/{train_label_collection}/{train_label_collection}_{i}/raster_labels.tif' 
               for i in train_ids]
source_paths = [f'{data_dir}/{source_collection}/{source_collection}_{i}/' 
               for i in train_ids]


# In[9]:


train_data = pd.DataFrame(np.array([train_ids, field_paths, label_paths, source_paths]).transpose(), 
                          columns=['folder_id', 'field_path', 'label_path', 'source_path'])
train_data.head()


# ## Extract band mean and std

# In[10]:


source_paths = [fn for fn in os.listdir(os.path.join(data_dir, 
                                                     source_collection)) 
if not fn.endswith('json')]


# In[11]:


means = []
stds = []
for source_path in tqdm(source_paths):
    m = []
    s = []
    for band in BAND_NAMES:
        with rasterio.open(os.path.join(data_dir, source_collection, source_path) + rf"/{band}.tif") as src:
            img = src.read()[0]
        m.append(np.mean(img))
        s.append(np.std(img))
    means.append(m)
    stds.append(s)
means = np.array(means).mean(0)
stds = np.array(stds).mean(0)


# 
# ## Extract field-crop data

# In[12]:


def extract_field_crop_data(data):
    field_ids = []
    crop_type = []
    field_area = []
    field_max_dim = []
    field_center_x = []
    field_center_y = []
    label_paths = []
    field_paths = []
    source_paths = []

    for i in tqdm(range(len(data))):
        with rasterio.open(data['field_path'].iloc[i]) as src:
            field_data = src.read()[0]
        if os.path.exists(data['label_path'].iloc[i]):
            with rasterio.open(data['label_path'].iloc[i]) as src:
                crop_data = src.read()[0]
        else:
            crop_data = None

        for field_id in np.unique(field_data)[1:]:
            ind = np.where(field_data == field_id)
            field_ids.append(field_id)
            field_area.append(len(ind[0]))
            field_max_dim.append(np.max(np.array(ind).max(1) - np.array(ind).min(1) + 1))
            field_center_y.append(np.mean(ind[0]))
            field_center_x.append(np.mean(ind[1]))
            field_paths.append(data['field_path'].iloc[i])
            source_paths.append(data['source_path'].iloc[i])
            if crop_data is not None:
                crop_type.append(np.unique(crop_data[ind])[-1])
                label_paths.append(data['label_path'].iloc[i])

    df = pd.DataFrame(np.array([field_ids,field_area, 
                              field_max_dim, field_center_x, 
                              field_center_y]).transpose(),
                    columns=['field_id', 'field_area', 
                            'field_max_dim', 'center_x', 'center_y'])
    df['field_path'] = field_paths
    df['source_path'] = source_paths
    if len(crop_type) > 0:
        df['crop_type'] = crop_type
        df['label_path'] = label_paths
    return df


# In[13]:


df = extract_field_crop_data(train_data)


# ## Relabel crop labels in sequential order

# In[14]:


crop_labels = np.unique(df['crop_type'])
df['crop_ind'] = df['crop_type'].apply(lambda x: np.where(crop_labels == x)[0][0])


# In[15]:


(df['crop_type'] == crop_labels[df['crop_ind']]).unique()


# ### Split the data into train and validation

# In[16]:


val_fraction = 0.2
random_seed = 42


# In[17]:


np.random.seed(random_seed)
df_train = []
df_val = []
for crop in df['crop_type'].unique():
    cur_df = df[df['crop_type'] == crop].reset_index(drop=True)
    unique_field_ids = cur_df['field_id'].unique()
    ind = np.arange(len(unique_field_ids))
    np.random.shuffle(ind)
    n_val = int(round(val_fraction * len(ind)))
    df_val.append(cur_df[cur_df['field_id'].isin(unique_field_ids[ind[:n_val]])])
    df_train.append(cur_df[cur_df['field_id'].isin(unique_field_ids[ind[n_val:]])])
df_train = pd.concat(df_train, ignore_index=True)
df_val = pd.concat(df_val, ignore_index=True)


# ## Exclude fields with area < 5 pixels

# In[18]:


df_train = df_train[df_train['field_area'] >= 5].reset_index(drop=True)
df_val = df_val[df_val['field_area'] >= 5].reset_index(drop=True)


# ## Crop and normalize the data

# In[19]:


patch_size = 16


# In[20]:


def crop_one_field(df, i, means, stds, size=16):
    imgs = []
    for band in BAND_NAMES:
        source_fn = rf"{df.iloc[i]['source_path']}{band}.tif"
        with rasterio.open(source_fn) as src:
            imgs.append(src.read()[0].astype(np.float64))
    imgs = np.array(imgs)
    
    field_id = df.iloc[i]['field_id']

    with rasterio.open(df.iloc[i]['field_path']) as src:
        fields = src.read()[0].astype(np.int64)
        imgs = np.concatenate([imgs, fields.reshape((1,) + fields.shape)])

    # z-scoring of the data
    imgs[:len(means)] = (imgs[:len(means)] - means.reshape(-1,1,1)) / stds.reshape(-1,1,1)
   
    # pad the image
    hs = int(size/2)
    imgs = np.pad(imgs, ((0, 0), (hs, hs), (hs, hs)))
    imgs[-1] = np.where(imgs[-1] == field_id, 1, 0)

    # crop 
    ind = [int(df.iloc[i]['center_y'] + hs), 
           int(df.iloc[i]['center_x'] + hs)]
    imgs = imgs[:, ind[0]-hs:ind[0]+hs, ind[1]-hs:ind[1]+hs]
    io.imsave(df['data_path'].iloc[i], imgs)


# In[21]:


def save_cropped(df, means, stds, size):
    for i in tqdm(range(len(df))):
        if not os.path.exists(df['data_path'].iloc[i]):
            crop_one_field(df, i, means, stds, size=size)


# In[22]:


cropped_path = path + 'data_cropped'
os.makedirs(cropped_path, exist_ok=True)


# In[23]:


def add_cropped_path(df):
    df['data_path'] = [rf"{cropped_path}/{df['field_id'].iloc[i]}_"
                       rf"{df['source_path'].iloc[i].split('_')[-1].rstrip('/')}.tif" 
                       for i in range(len(df))]
    return df


# In[24]:


df_val = add_cropped_path(df_val)
df_train = add_cropped_path(df_train)


# In[25]:


save_cropped(df_val, means, stds, patch_size)
save_cropped(df_train,  means, stds, patch_size)


# ## Prepare the test dataset

# In[26]:


test_paths = os.listdir(os.path.join(data_dir, test_label_collection))
test_ids = [fn.split('_')[-1] for fn in test_paths if 'labels_test' in fn]

field_paths = [f'{data_dir}/{test_label_collection}/{test_label_collection}_{i}/field_ids.tif' 
               for i in test_ids]
label_paths = [f'{data_dir}/{test_label_collection}/{test_label_collection}_{i}/raster_labels.tif' 
               for i in test_ids]
source_paths = [f'{data_dir}/{source_collection}/{source_collection}_{i}/' 
               for i in test_ids]


# In[27]:


test_data = pd.DataFrame(np.array([test_ids, field_paths, label_paths, source_paths]).transpose(), 
                          columns=['folder_id', 'field_path', 'label_path', 'source_path'])
test_data.head()


# In[28]:


df_test = extract_field_crop_data(test_data)
df_test['crop_ind'] = 0
df_test['crop_type'] = 1


# In[29]:


df_test = add_cropped_path(df_test)


# In[30]:


save_cropped(df_test, means, stds, patch_size)


# ## Save train, validation, and test dataframes

# In[31]:


df_train.to_csv(path + 'df_train.csv', index=False)
df_val.to_csv(path + 'df_val.csv', index=False)
df_test.to_csv(path + 'df_test.csv', index=False)


# In[31]:




