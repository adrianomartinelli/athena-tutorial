"""
Load MIBI data from the provided data sources:
- https://www.angelolab.com/mibi-data
- https://mibi-share.ionpath.com/tracker/imageset

Notes:
    - cellData.csv: Contains PatientIDs 42,43,44 for which no tiff stack exists
    - cellData.csv: PatientID 30 missing
    - cellSata.csv: for some observations I cannot find the layer in the tiff stack (CD163, CSF-1R,...)
"""

# %%
from spatialOmics import SpatialOmics
from pathlib import Path
import pandas as pd
import numpy as np

root = Path('/Users/art/Library/CloudStorage/Box-Box/documents/internship_adriano/MIBI-data/')
dir_cm = Path('TNBC_shareCellData')
dir_expr = Path('Keren')

fmeta = 'cellData.csv'
fpatient = 'patient_class.csv'

# %% load meta data from tiff stack
from PIL import Image, ImageSequence
from PIL.TiffTags import TAGS
import re

f = Path(root / dir_expr / 'TA459_multipleCores2_Run-4_Point34.tiff')
im = Image.open(f)

img = []
_meta = []
channel = []
regex = re.compile('"channel.target": "(.*)",', re.IGNORECASE)
for i, page in enumerate(ImageSequence.Iterator(im)):
    img.append(page)

    meta_dict = {TAGS[key]: page.tag[key] for key in page.tag}
    _meta.append(pd.DataFrame.from_dict(meta_dict))

    s = meta_dict['ImageDescription'][0]
    m = regex.search(s)
    channel.append(m.group(1))

channels = pd.DataFrame(channel, columns=['channel'])
channels['stack_index'] = range(len(channel))

# %%

df = pd.read_csv(root / dir_cm / fpatient, header=None)
meta = pd.read_csv(root / dir_cm / fmeta)

feat_expr = channels.channel.values

meta = meta.rename({'cellLabelInImage': 'cell_id'}, axis=1)
meta.loc[:, 'SampleID'] = meta.SampleID.astype(str)

feat_cat = ['SampleID', 'tumorYN', 'tumorCluster', 'Group', 'immuneCluster', 'immuneGroup']
for i in feat_cat:
    meta.loc[:, i] = meta[i].astype('category')
meta = meta.set_index('cell_id')

df.columns = ['SampleID', 'class']
df.loc[:, 'SampleID'] = df.SampleID.astype(str)

s1, s2 = set(meta.SampleID.unique()), set(df.SampleID)
s = s1.intersection(s2)  # only keep SampleIDs for which we have all information
# s = s.union(s1)
spl = pd.DataFrame(s, columns=['SampleID'])
spl = pd.merge(spl, df, how='left', on='SampleID')

spl.loc[:,'cell_mask_file'] = [f'p{i}_labeledcellData.tiff' for i in spl.SampleID.unique()]
spl.loc[:,'tiff_stack_file'] = [f'TA459_multipleCores2_Run-4_Point{i}.tiff' for i in spl.SampleID.unique()]

channels = channels.set_index('channel')
obs = {i: meta.loc[meta.SampleID == i, ~meta.columns.isin(feat_expr)] for i in spl.SampleID}
X = {i: meta.loc[meta.SampleID == i, feat_expr] for i in spl.SampleID}
var = {i: channels.loc[x.columns].reset_index().rename({'index':'channel'}, axis=1) for i,x in X.items()}

# %%
so = SpatialOmics()

# set spatialOmics attributes
so.X = X
so.spl = spl.set_index('SampleID')
so.obs = obs
so.var = var

# %%
for spl, fcm, fexpr in zip(so.spl.index, so.spl.cell_mask_file, so.spl.tiff_stack_file):
    try:
        # add high-dimensional tiff image
        so.add_mask(spl, 'cellmasks', root / dir_cm / fcm, to_store=False)

        # add segmentation mask
        so.add_image(spl, root / dir_expr / fexpr, to_store=False)
    except FileNotFoundError:
        print(f'File for sample {spl} not found.\n{fcm}\n{fexpr}')

# %% post-process cellmasks
for spl in so.spl.index:
    mask_ids_not_in_meta = set(np.unique(so.masks[spl]['cellmasks'])) - set(so.X[spl].index)
    for _id in mask_ids_not_in_meta:
        mask = so.masks[spl]['cellmasks']
        mask[mask == _id] = 0

# %% phenotype encoding
spls = so.spl.index
groups = set()
for spl in spls:
    obs = so.obs[spl]

    tgrp = [f't{i}' for i in obs.Group]
    igrp = [f'i{i}' for i in obs.immuneGroup]
    grp = [i+j for i,j in zip(tgrp, igrp)]
    groups = groups.union(set(grp))

    so.obs[spl]['group'] = grp
    so.obs[spl]['group'] = so.obs[spl]['group'].astype('category')

groups = list(groups)
groups = sorted(groups, key=lambda x: x.startswith('t2'))  # sort such that tumor groups come first
groups = pd.DataFrame(groups, columns = ['groups'])
groups['group_id'] = range(len(groups))

for spl in spls:
    so.obs[spl]['cell_type'] = so.obs[spl]['tumorYN'].map({0:'immune', 1:'tumor'}).astype('category')
    so.obs[spl]['cell_type_id'] = np.asarray(so.obs[spl]['tumorYN']) + 1
    so.obs[spl]['cell_type_id'] = so.obs[spl]['cell_type_id'].astype('category')

# %% generate colormaps
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
import colorcet as cc

so.uns['cmaps']['default'] = plt.get_cmap('plasma')

grp2id = groups.set_index('groups').group_id.to_dict()

for spl in spls:
    so.obs[spl]['group_id'] = [grp2id[i] for i in so.obs[spl].group]
    so.obs[spl]['group_id'] = so.obs[spl].group_id.astype('category')

# select colormaps for tumor/immune cells
tmap = sns.color_palette(cc.glasbey_warm, n_colors=5)
imap = sns.color_palette(cc.glasbey_cool, n_colors=12)
palette = tmap + imap

so.uns['cmap_labels']['group_id'] = {j:i for i,j in grp2id.items()}
so.uns['cmaps']['group_id'] = ListedColormap(palette)

cmap = ['white', 'darkgreen', 'darkred']
cmap_labels = {0: 'background', 1: 'immune',  2:'tumor'}
cmap = ListedColormap(cmap)
so.uns['cmaps'].update({'cell_type_id': cmap})
so.uns['cmap_labels'].update({'cell_type_id': cmap_labels})

cmap = ['darkgreen', 'darkred']
cmap_labels = {0: 'immune',  1:'tumor'}
cmap = ListedColormap(cmap)
so.uns['cmaps'].update({'cell_type': cmap})
so.uns['cmap_labels'].update({'cell_type': cmap_labels})

so.uns['cmaps'].update({'default': cm.plasma})
# %%
fsave = Path('mibi.h5py')
so.to_h5py(fsave)
reloaded = SpatialOmics.from_h5py(fsave)