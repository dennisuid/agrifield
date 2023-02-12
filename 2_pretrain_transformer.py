#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/amedyukhina/AgrifieldNet/blob/main/2_pretrain_transformer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


import os
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import nn
from skimage import io
from torchvision import transforms as torch_transforms
from torchvision import models
import pylab as plt
from skimage.segmentation import mark_boundaries
import rasterio
from radiant_mlhub import Dataset
import tarfile
import getpass


# ## Specify path, random seed, and band names

# In[4]:


path = 'agri_data/'
# path = 'gdrive/MyDrive/agrifield_data/'


# In[5]:


if 'gdrive' in path:
    from google.colab import drive
    drive.mount('/content/gdrive')


# In[6]:


random_seed = 42


# In[7]:


np.random.seed(random_seed)


# ## Load the data

# In[8]:


collection_name = 'ref_agrifieldnet_competition_v1'
BAND_NAMES = ['B01', 'B02', 'B03', 'B04','B05', 'B06', 'B07', 
              'B08','B8A', 'B09', 'B11', 'B12']
data_dir = path + 'data'

source_collection = f'{collection_name}_source'
train_label_collection = f'{collection_name}_labels_train'
test_label_collection = f'{collection_name}_labels_test'


# In[9]:


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

# In[10]:


train_paths = os.listdir(os.path.join(data_dir, train_label_collection))
train_ids = [fn.split('_')[-1] for fn in train_paths if 'labels_train' in fn]

field_paths = [f'{data_dir}/{train_label_collection}/{train_label_collection}_{i}/field_ids.tif' 
               for i in train_ids]
label_paths = [f'{data_dir}/{train_label_collection}/{train_label_collection}_{i}/raster_labels.tif' 
               for i in train_ids]
source_paths = [f'{data_dir}/{source_collection}/{source_collection}_{i}/' 
               for i in train_ids]


# In[11]:


train_data = pd.DataFrame(np.array([train_ids, field_paths, label_paths, source_paths]).transpose(), 
                          columns=['folder_id', 'field_path', 'label_path', 'source_path'])
train_data.head()


# ## Read mean and std

# In[12]:


source_paths = [fn for fn in os.listdir(os.path.join(data_dir, 
                                                     source_collection)) 
if not fn.endswith('json')]


# In[ ]:


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


# ## Extract field-crop data

# In[ ]:


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
      with rasterio.open(data['label_path'].iloc[i]) as src:
          crop_data = src.read()[0]

      for field_id in np.unique(field_data)[1:]:
          ind = np.where(field_data == field_id)
          field_ids.append(field_id)
          crop_type.append(np.unique(crop_data[ind])[-1])
          field_area.append(len(ind[0]))
          field_max_dim.append(np.max(np.array(ind).max(1) - np.array(ind).min(1) + 1))
          field_center_y.append(np.mean(ind[0]))
          field_center_x.append(np.mean(ind[1]))
          label_paths.append(data['label_path'].iloc[i])
          field_paths.append(data['field_path'].iloc[i])
          source_paths.append(data['source_path'].iloc[i])

  df = pd.DataFrame(np.array([field_ids, crop_type, field_area, 
                              field_max_dim, field_center_x, 
                              field_center_y]).transpose(),
                    columns=['field_id', 'crop_type', 'field_area', 
                            'field_max_dim', 'center_x', 'center_y'])
  df['label_path'] = label_paths
  df['field_path'] = field_paths
  df['source_path'] = source_paths
  return df


# In[ ]:


df = extract_field_crop_data(train_data)


# ## Extract field masks

# In[ ]:


patch_size = 16


# In[ ]:


def crop_one_field(df, i, size=16):
    
    field_id = df.iloc[i]['field_id']

    with rasterio.open(df.iloc[i]['field_path']) as src:
        fields = src.read()[0].astype(np.int64)
   
    # pad the image
    hs = int(size/2)
    fields = np.pad(fields, ((hs, hs), (hs, hs)))
    fields = np.where(fields == field_id, 1, 0)

    # crop 
    ind = [int(df.iloc[i]['center_y'] + hs), 
           int(df.iloc[i]['center_x'] + hs)]
    field = fields[ind[0]-hs:ind[0]+hs, ind[1]-hs:ind[1]+hs]
    return field


# In[ ]:


field_masks = np.array([crop_one_field(df, i, size=patch_size) for i in tqdm(range(len(df)))])
field_masks.shape


# In[ ]:


s = 3
fig, axes = plt.subplots(6, 5, figsize=(s*5, s*6))
for ax, img in zip(axes.ravel(), field_masks[:30]):
  plt.sca(ax)
  io.imshow(img)


# ## Specify Data Loading pipeline

# ### Specify dataset and transforms

# In[ ]:


train_transforms = torch_transforms.Compose([
        torch_transforms.RandomHorizontalFlip(),
        torch_transforms.RandomVerticalFlip(),
        torch_transforms.RandomRotation(degrees=180),
    ])


# In[ ]:


class AgriDataset(torch.utils.data.Dataset):

  def __init__(self, df, transforms=None, size=16, pos_thr=5, neg_thr=30,
               means=None, stds=None):
      self.df = df
      self.transforms = transforms
      self.size = size
      self.pos_thr = pos_thr
      self.neg_thr = neg_thr
      self.stds = stds
      self.means = means

  def __getitem__(self, index):
    imgs = []
    for band in BAND_NAMES:
        source_fn = rf"{self.df.iloc[index]['source_path']}{band}.tif"
        with rasterio.open(source_fn) as src:
          imgs.append(src.read()[0].astype(np.float64))
    
    imgs = np.array(imgs)

    # z-scoring of the data
    if self.means is not None and self.stds is not None:
        imgs[:len(self.means)] = (imgs[:len(self.means)] - self.means.reshape(-1,1,1)) / self.stds.reshape(-1,1,1)
    else:
        imgs[:len(self.means)] = imgs[:len(self.means)] / 255.

    # crop ancor image
    hs = int(self.size/2)
    ind_ancor = np.random.randint(hs, imgs.shape[-1] - hs, 2)
    img_ancor = imgs[:, ind_ancor[0]-hs:ind_ancor[0]+hs, ind_ancor[1]-hs:ind_ancor[1]+hs]

    # crop positive pair
    inds = np.array([np.random.randint(ind-self.pos_thr, ind+self.pos_thr) for ind in ind_ancor])
    inds = np.min(np.stack([inds, np.ones_like(inds)*(imgs.shape[-1] - hs)]), axis=0)
    inds = np.max(np.stack([inds, np.ones_like(inds)*hs]), axis=0)
    img_pos = imgs[:, inds[0]-hs:inds[0]+hs, inds[1]-hs:inds[1]+hs]

    # crop negative pair
    while np.sqrt(np.sum((inds - ind_ancor)**2)) < self.neg_thr:
        inds = np.random.randint(hs, imgs.shape[-1] - hs, 2)
    img_neg = imgs[:, inds[0]-hs:inds[0]+hs, inds[1]-hs:inds[1]+hs]

    # get random field masks
    ind = np.random.randint(0, len(field_masks), 3)
    masks = field_masks[ind]
      
    # apply transforms
    if self.transforms:
        imgs = torch.stack([self.transforms(torch.tensor(img).float()) for img in [img_ancor, img_pos, img_neg]])
        masks = torch.stack([self.transforms(torch.tensor(mask).unsqueeze(0).float()).squeeze(0) for mask in masks])

    return imgs.float(), masks.float()
    

  def __len__(self):
    return len(self.df)


# ### Test the loaders

# In[ ]:


batch_size = 8


# In[ ]:


np.random.seed(random_seed)
torch.manual_seed(random_seed)
dl_train = torch.utils.data.DataLoader(
    AgriDataset(df, means=means, stds=stds,
                transforms=train_transforms,
                ), 
    shuffle=True, batch_size=batch_size, num_workers=2,
    )


# In[ ]:


torch.manual_seed(random_seed)
imgs, masks = next(iter(dl_train))


# In[ ]:


imgs.shape, masks.shape


# In[ ]:


ind = 0
s = 3


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(s*3, s*2))
for i in range(3):
  plt.sca(axes[0, i])
  img = imgs[ind][i][np.array([3,2,1])].numpy().transpose(1,2,0)
  io.imshow(img)
  plt.sca(axes[1, i])
  io.imshow(masks[ind][i].numpy())


# ### Specify the loaders

# In[ ]:


batch_size = 64


# In[ ]:


np.random.seed(random_seed)
torch.manual_seed(random_seed)

dl_train = torch.utils.data.DataLoader(
    AgriDataset(df, means=means, stds=stds,
                transforms=train_transforms,
                ), 
    shuffle=True, batch_size=batch_size, num_workers=2,
    )


# ## Specify the training pipeline

# ### Model and loss

# In[ ]:


import math

class PositionEmbeddingSine(nn.Module):
    """
    Adapted from https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        not_mask = torch.ones((x.shape[0],) + x.shape[-2:])
        y_embed = not_mask.cumsum(1, dtype=torch.float32).to(x.device)
        x_embed = not_mask.cumsum(2, dtype=torch.float32).to(x.device)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# In[ ]:


class Transformer(nn.Module):

    def __init__(self, in_channels, hidden_dim, nheads=8, num_layers=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, 
                                                   nhead=nheads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_layers)
        self.input_embed = nn.Conv2d(in_channels, hidden_dim, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(32, 64, bias=True),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128, bias=True),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64, bias=True),
        )

        # spatial positional encodings
        self.position_encodings = PositionEmbeddingSine(torch.div(hidden_dim, 
                                                                  2, 
                                                                  rounding_mode='trunc'),
                                                        normalize=True)

    def forward(self, x, mask):
        x = self.input_embed(x)
        mask = (1 - mask).flatten(-2, -1)
        mask[:,0] = 0
        pos = self.position_encodings(x)
        out = self.transformer_encoder((pos + x).flatten(-2, -1).transpose(-1, -2),
                                       src_key_padding_mask=mask)
        return out[:,0]


# In[ ]:


net = Transformer(in_channels=len(BAND_NAMES), hidden_dim=len(BAND_NAMES), nheads=4).cuda()


# In[ ]:


loss_fn = nn.TripletMarginLoss().cuda()


# ### Optimizer and scheduler

# In[ ]:


lr = 0.001
weight_decay = 0.05
epochs = 500
patience = 10
factor = 0.1
gamma = 0.9


# In[ ]:


optimizer = torch.optim.AdamW(
    params=[{"params": [p for p in net.parameters() if p.requires_grad]}],
            lr=lr, weight_decay=weight_decay
)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=factor, patience=patience
    )
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
#     optimizer, gamma=gamma
#     )


# ### Training step

# In[ ]:


def train_epoch(net, loss_fn, dl_train, optimizer, lr_scheduler):
  net.train()
  loss_fn.train()
  epoch_loss = 0
  step = 0
  for imgs, masks in tqdm(dl_train):
    step += 1
    optimizer.zero_grad()
    outputs = [net(imgs[:,i].cuda(), masks[:,i].cuda()) for i in range(imgs.shape[1])]
    losses = loss_fn(*outputs)
    losses.backward()
    optimizer.step()
    epoch_loss += losses.item()
    # lr_scheduler.step()
  epoch_loss /= step
  return epoch_loss


# ## Train the model

# In[ ]:


model_name = rf'{int(time.time())}_{time.ctime()}_pretraining'
model_dir = os.path.join(path, 'models', model_name)
os.makedirs(model_dir, exist_ok=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'torch.manual_seed(random_seed)\nnp.random.seed(random_seed)\nbest_loss = 10**10\ntrain_losses = []\nval_losses = []\nfor epoch in range(epochs):\n    # training pass\n    train_loss = train_epoch(net, loss_fn, dl_train, optimizer, lr_scheduler)\n    lr_scheduler.step(train_loss)\n\n    # output\n    print(f"epoch {epoch + 1} training loss: {train_loss:.4f}; "\n          f"lr: {optimizer.param_groups[0][\'lr\']}")\n    train_losses.append(train_loss)\n    if train_loss <= best_loss:\n      best_loss = train_loss\n      torch.save(net.state_dict(), os.path.join(model_dir, \'best_model.pth\'))\n      print(\'saving best model\')\n    torch.save(net.state_dict(), os.path.join(model_dir, \'last_model.pth\'))')


# In[ ]:


plt.figure(figsize=(10, 5))
plt.plot(train_losses,'r')


# In[ ]:




